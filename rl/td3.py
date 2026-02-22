from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from configs import TD3Config

from .base import BaseRL
from .transition import TransitionBatch


class TD3(BaseRL):
    def __init__(
        self,
        gamma: float,
        state_dim: tuple[int, ...],
        action_dim: tuple[int, ...],
        reward_dim: tuple[int, ...],
        actor_net: type[nn.Module],
        critic_net: type[nn.Module],
        lr: float,
        tau: float,
        weight_decay: float,
        policy_update_freq: int = 2,
        policy_noise: float = 0.2,
        noise_clip: float = 0.5,
        max_action: float = 1.0,
        reward_weight: torch.Tensor = torch.ones(1),
        device: Optional[torch.device] = None,
        use_jit: bool = True,
        train: bool = True,
        name: str = "TD3",
        load_chkpt: bool = False,
        chkpt_file: Optional[Path] = None,
    ):
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim

        self.lr = lr
        self.tau = tau
        self.weight_decay = weight_decay
        self.policy_update_freq = policy_update_freq
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.max_action = max_action
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.reward_weight = reward_weight.to(self.device)
        self.use_jit = use_jit
        self.name = name

        self.actor = actor_net(self.state_dim, self.action_dim)
        self.actor_target = actor_net(self.state_dim, self.action_dim)

        self.Q1 = critic_net(self.state_dim, self.action_dim, self.reward_dim)
        self.Q2 = critic_net(self.state_dim, self.action_dim, self.reward_dim)
        self.Q1_target = critic_net(self.state_dim, self.action_dim, self.reward_dim)
        self.Q2_target = critic_net(self.state_dim, self.action_dim, self.reward_dim)

        self.to(self.device)
        self._hard_update()
        if self.use_jit:
            self._jit_compile()

        self.actor_optim = optim.Adam(
            self.actor.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.Q_optim = optim.Adam(
            list(self.Q1.parameters()) + list(self.Q2.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.total_steps = 0

        self.train(train)
        if load_chkpt and chkpt_file:
            self.load(chkpt_file)

    def _jit_compile(self):
        self.actor = torch.jit.script(self.actor)
        self.Q1 = torch.jit.script(self.Q1)
        self.Q2 = torch.jit.script(self.Q2)

        self.actor_target = torch.jit.script(self.actor_target)
        self.Q1_target = torch.jit.script(self.Q1_target)
        self.Q2_target = torch.jit.script(self.Q2_target)

    @torch.no_grad()
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        action = self.actor(state)
        return action

    @torch.no_grad()
    def _hard_update(self) -> None:
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())

        self.actor_target.eval()
        self.Q1_target.eval()
        self.Q2_target.eval()

    @torch.no_grad()
    def _soft_update(self) -> None:
        # actor
        for target_param, source_param in zip(
            self.actor_target.parameters(), self.actor.parameters()
        ):
            target_param.data.lerp_(source_param.data, self.tau)
        # Q1
        for target_param, source_param in zip(
            self.Q1_target.parameters(), self.Q1.parameters()
        ):
            target_param.data.lerp_(source_param.data, self.tau)
        # Q2
        for target_param, source_param in zip(
            self.Q2_target.parameters(), self.Q2.parameters()
        ):
            target_param.data.lerp_(source_param.data, self.tau)

    def update(self, transition: TransitionBatch) -> dict[str, Any]:
        metrics = {}
        critic_metrics = self._update_critic(transition)
        metrics["critic"] = critic_metrics

        if self.total_steps % self.policy_update_freq == 0:
            actor_metrics = self._update_actor(transition)
            metrics["actor"] = actor_metrics

        self._soft_update()
        self.total_steps += 1

        return metrics

    def _update_critic(self, transition: TransitionBatch) -> dict[str, Any]:
        s, a, r, ns, done = transition.unpack()
        mask = (~done).unsqueeze(1)

        with torch.no_grad():
            noise = torch.randn_like(a) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = self.actor_target(ns) + noise
            next_action = next_action.clamp(-self.max_action, self.max_action)

            q1_next = self.Q1_target(ns, next_action)
            q2_next = self.Q2_target(ns, next_action)
            q_next_min = torch.min(q1_next, q2_next)
            q_target = r + self.gamma * mask * q_next_min

        q1_pred = self.Q1(s, a)
        q2_pred = self.Q2(s, a)

        loss_q1 = F.mse_loss(q1_pred, q_target, reduction="none")
        loss_q2 = F.mse_loss(q2_pred, q_target, reduction="none")

        weights = transition.weights
        if weights is not None:
            loss_q1 = loss_q1 * weights
            loss_q2 = loss_q2 * weights

        critic_loss = 0.5 * (loss_q1.mean() + loss_q2.mean())

        self.Q_optim.zero_grad()
        critic_loss.backward()
        self.Q_optim.step()

        with torch.no_grad():
            td_error = torch.abs(q1_pred - q_target)

        return {
            "critic_loss": critic_loss.detach().cpu().item(),
            "q1_pred": q1_pred.detach().cpu(),
            "q2_pred": q2_pred.detach().cpu(),
            "q_target": q_target.detach().cpu(),
            "td_error": td_error.detach().cpu(),
        }

    def _update_actor(self, transition: TransitionBatch) -> dict[str, Any]:
        s, a, r, ns, done = transition.unpack()

        action = self.actor(s)
        q1 = self.Q1(s, action)
        q2 = self.Q2(s, action)
        q = torch.min(q1, q2)
        q = torch.matmul(q, self.reward_weight)

        actor_loss = -q.mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        return {
            "actor_loss": actor_loss.detach().cpu().item(),
            "q": q.detach().cpu(),
        }

    def save(self, ckpt_file: Path) -> None:
        torch.save(
            {
                # networks
                "actor": self.actor.state_dict(),
                "Q1": self.Q1.state_dict(),
                "Q2": self.Q2.state_dict(),
                # targets
                "actor_target": self.actor_target.state_dict(),
                "Q1_target": self.Q1_target.state_dict(),
                "Q2_target": self.Q2_target.state_dict(),
                # optimizers
                "actor_optimizer": self.actor_optim.state_dict(),
                "Q_optimizer": self.Q_optim.state_dict(),
                "total_steps": self.total_steps,
            },
            ckpt_file,
        )

    def load(self, ckpt_file: Path) -> None:
        ckpt = torch.load(ckpt_file, map_location=self.device)

        self.actor.load_state_dict(ckpt["actor"])
        self.Q1.load_state_dict(ckpt["Q1"])
        self.Q2.load_state_dict(ckpt["Q2"])

        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.Q1_target.load_state_dict(ckpt["Q1_target"])
        self.Q2_target.load_state_dict(ckpt["Q2_target"])

        self.actor_optim.load_state_dict(ckpt["actor_optimizer"])
        self.Q_optim.load_state_dict(ckpt["Q_optimizer"])

        self.total_steps = ckpt["total_steps"]

    def log_params(self, *, prefix: str = "agent/") -> None:
        if not mlflow.active_run():
            raise RuntimeError(
                "No active MLflow run found. Call mlflow.start_run() first."
            )

        p = prefix
        mlflow.log_params(
            {
                f"{p}name": self.name,
                f"{p}gamma": self.gamma,
                f"{p}state_dim": self.state_dim,
                f"{p}action_dim": self.action_dim,
                f"{p}reward_dim": self.reward_dim,
                f"{p}lr": self.lr,
                f"{p}tau": self.tau,
                f"{p}weight_decay": self.weight_decay,
                f"{p}policy_update_freq": self.policy_update_freq,
                f"{p}policy_noise": self.policy_noise,
                f"{p}noise_clip": self.noise_clip,
                f"{p}max_action": self.max_action,
                f"{p}reward_weight": self.reward_weight.detach().cpu().tolist(),
                f"{p}device": self.device,
                f"{p}use_jit": self.use_jit,
            }
        )

    @classmethod
    def from_config(cls, cfg: TD3Config) -> "TD3":
        return cls(
            gamma=cfg.gamma,
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            reward_dim=cfg.reward_dim,
            actor_net=cfg.actor_net,
            critic_net=cfg.critic_net,
            lr=cfg.lr,
            tau=cfg.tau,
            weight_decay=cfg.weight_decay,
            policy_update_freq=cfg.policy_update_freq,
            policy_noise=cfg.policy_noise,
            noise_clip=cfg.noise_clip,
            max_action=cfg.max_action,
            reward_weight=cfg.reward_weight,
            device=cfg.device,
            use_jit=cfg.use_jit,
            train=cfg.train,
            name=cfg.name,
            load_chkpt=cfg.load_chkpt,
            chkpt_file=cfg.chkpt_file,
        )

    def to(self, device: torch.device, non_blocking: bool = True) -> None:
        self.actor.to(device, non_blocking=non_blocking)
        self.Q1.to(device, non_blocking=non_blocking)
        self.Q2.to(device, non_blocking=non_blocking)

        self.actor_target.to(device, non_blocking=non_blocking)
        self.Q1_target.to(device, non_blocking=non_blocking)
        self.Q2_target.to(device, non_blocking=non_blocking)

    def train(self, mode: bool = True) -> None:
        self.actor.train(mode)
        self.Q1.train(mode)
        self.Q2.train(mode)

        super().train(mode)

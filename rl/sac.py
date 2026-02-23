from pathlib import Path
from typing import Any, Optional

import mlflow
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from analysis import MlflowWriter, TBWriter
from configs import SACConfig

from .base import BaseRL
from .transition import TransitionBatch

# https://github.com/rail-berkeley/softlearning/blob/master/softlearning/algorithms/sac.py
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/sac.py

# TODO - add lr scheduler
# TODO - support shared base network between actor and critics


def default_target_entropy(action_space: tuple[int, ...]) -> int:
    import operator
    from functools import reduce

    return -reduce(operator.mul, action_space, 1)


class SAC(BaseRL):
    def __init__(
        self,
        # environment
        gamma: float,
        state_dim: tuple[int, ...],
        action_dim: tuple[int, ...],
        reward_dim: tuple[int, ...],
        # actor
        log_std_min: float,
        log_std_max: float,
        actor_net: type[nn.Module],
        critic_net: type[nn.Module],
        # training
        lr: float,
        tau: float,
        target_entropy: float,
        weight_decay: float,
        policy_update_freq: int = 1,
        # etc
        reward_weight: torch.Tensor = torch.ones(1),
        device: Optional[torch.device] = None,
        use_jit: bool = True,
        train: bool = True,
        name: str = "SAC",
        # load model
        load_ckpt: bool = False,
        ckpt_file: Optional[Path] = None,
    ):
        super().__init__(device, gamma)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim

        self.lr = lr
        self.tau = tau
        self.weight_decay = weight_decay
        self.policy_update_freq = policy_update_freq
        self.reward_weight = reward_weight.to(self.device)
        self.use_jit = use_jit
        self.name = name
        self.total_steps = 0

        # alpha
        self.target_entropy = target_entropy
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.lr)

        # Network Initialization
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.actor = actor_net(
            self.state_dim, self.action_dim, self.log_std_min, self.log_std_max
        )

        self.Q1 = critic_net(self.state_dim, self.action_dim, self.reward_dim)
        self.Q2 = critic_net(self.state_dim, self.action_dim, self.reward_dim)
        self.Q1_target = critic_net(self.state_dim, self.action_dim, self.reward_dim)
        self.Q2_target = critic_net(self.state_dim, self.action_dim, self.reward_dim)
        self.to(self.device)
        self._hard_update()
        if self.use_jit:
            self._jit_compile()

        # Network Optimizers
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.Q_optim = torch.optim.Adam(
            list(self.Q1.parameters()) + list(self.Q2.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.train(train)
        if load_ckpt and ckpt_file:
            self.load(ckpt_file)

    def _jit_compile(self):
        # sampling used in actor is not allowed on jit compilation
        # self.actor = torch.jit.script(self.actor)
        self.Q1 = torch.jit.script(self.Q1)
        self.Q2 = torch.jit.script(self.Q2)
        self.Q1_target = torch.jit.script(self.Q1_target)
        self.Q2_target = torch.jit.script(self.Q2_target)

    def get_alpha(self):
        return self.log_alpha.exp().item()

    @torch.no_grad()
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        action, _ = self.actor(state, deterministic=not self._train)

        return action

    @torch.no_grad()
    def _hard_update(self) -> None:
        # copy weights
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())
        # set eval mode
        self.Q1_target.eval()
        self.Q2_target.eval()

    @torch.no_grad()
    def _soft_update(self) -> None:
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
        alpha_metrics = self._update_alpha(transition)
        metrics["alpha"] = alpha_metrics
        critic_metrics = self._update_critic(transition)
        metrics["critic"] = critic_metrics

        if self.total_steps % self.policy_update_freq == 0:
            actor_metrics = self._update_actor(transition)
            metrics["actor"] = actor_metrics

        self._soft_update()
        self.total_steps += 1

        return metrics

    def _update_alpha(self, transition: TransitionBatch) -> dict[str, float]:
        s, a, r, ns, done = transition.unpack()
        new_action, log_prob = self.actor(s)

        # alpha loss = -alpha*log_policy -alpha*target_entropy = -alpha*(log_prob+target_entropy)
        entropy = log_prob.detach().mean()
        # alpha_loss = self.log_alpha.exp() * (entropy + self.target_entropy) # softlearning
        alpha_loss = -self.log_alpha * (entropy + self.target_entropy)  # SB3

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        return {
            "alpha": self.get_alpha(),
            "sample_entropy": (-entropy).cpu().item(),
            "alpha_loss": alpha_loss.detach().cpu().item(),
        }

    def _update_critic(self, transition: TransitionBatch) -> dict[str, Any]:
        s, a, r, ns, done = transition.unpack()

        # alpha is constant
        alpha = self.log_alpha.detach().exp()
        mask = (~done).unsqueeze(1)

        # q function loss
        # Q target = r + gamma * (minQ(ns, a~) - alpha * log_pi(ns))
        with torch.no_grad():
            next_action, next_log_prob = self.actor(ns)
            q1_next = self.Q1_target(ns, next_action)
            q2_next = self.Q2_target(ns, next_action)
            q_next_min = torch.min(q1_next, q2_next)
            q_target = r + self.gamma * mask * (q_next_min - alpha * next_log_prob)

        q1_pred = self.Q1(s, a)
        q2_pred = self.Q2(s, a)
        critic_loss = 0.5 * (
            F.mse_loss(q1_pred, q_target) + F.mse_loss(q2_pred, q_target)
        )

        self.Q_optim.zero_grad()
        critic_loss.backward()
        self.Q_optim.step()

        return {
            "critic_loss": critic_loss.detach().cpu().item(),
            "q1_pred": q1_pred.detach().cpu(),
            "q2_pred": q2_pred.detach().cpu(),
            "q_target": q_target.detach().cpu(),
        }

    def _update_actor(self, transition: TransitionBatch) -> dict[str, Any]:
        s, a, r, ns, done = transition.unpack()
        alpha = self.log_alpha.detach().exp()

        action, log_prob = self.actor(s)
        q1 = self.Q1(s, action)
        q2 = self.Q2(s, action)
        q = torch.min(q1, q2)  # softlearning
        # q = 0.5 * (q1 + q2)  # SB3
        q = torch.matmul(q, self.reward_weight)

        actor_loss = (alpha * log_prob - q).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        return {
            "actor_loss": actor_loss.detach().cpu().item(),
            "log_prob": log_prob.detach().cpu(),
            "q": q.detach().cpu(),
        }

    def save(self, ckpt_file):
        torch.save(
            {
                # networks
                "actor": self.actor.state_dict(),
                "Q1": self.Q1.state_dict(),
                "Q2": self.Q2.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                # targets
                "Q1_target": self.Q1_target.state_dict(),
                "Q2_target": self.Q2_target.state_dict(),
                # optimizers
                "actor_optim": self.actor_optim.state_dict(),
                "Q_optim": self.Q_optim.state_dict(),
                "alpha_optim": self.alpha_optim.state_dict(),
                "total_steps": self.total_steps,
            },
            ckpt_file,
        )

    def load(self, ckpt_file):
        ckpt = torch.load(ckpt_file, map_location=self.device)

        self.actor.load_state_dict(ckpt["actor"])
        self.Q1.load_state_dict(ckpt["Q1"])
        self.Q2.load_state_dict(ckpt["Q2"])
        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))

        self.Q1_target.load_state_dict(ckpt["Q1_target"])
        self.Q2_target.load_state_dict(ckpt["Q2_target"])

        self.actor_optim.load_state_dict(ckpt["actor_optim"])
        self.Q_optim.load_state_dict(ckpt["Q_optim"])
        self.alpha_optim.load_state_dict(ckpt["alpha_optim"])

        self.total_steps = ckpt["total_steps"]

    def log_params(
        self, mlflow_writer: MlflowWriter, *, prefix: str = "agent/"
    ) -> None:
        p = prefix

        params_dict = {
            f"{p}name": self.name,
            f"{p}gamma": self.gamma,
            f"{p}state_dim": self.state_dim,
            f"{p}action_dim": self.action_dim,
            f"{p}reward_dim": self.reward_dim,
            f"{p}log_std_min": self.log_std_min,
            f"{p}log_std_max": self.log_std_max,
            f"{p}lr": self.lr,
            f"{p}tau": self.tau,
            f"{p}target_entropy": self.target_entropy,
            f"{p}weight_decay": self.weight_decay,
            f"{p}policy_update_freq": self.policy_update_freq,
            f"{p}reward_weight": self.reward_weight.detach().cpu().tolist(),
            f"{p}device": str(self.device),
            f"{p}use_jit": self.use_jit,
        }

        mlflow_writer.log_params(params_dict)

    @classmethod
    def from_config(cls, cfg: SACConfig) -> "SAC":
        return cls(
            gamma=cfg.gamma,
            state_dim=cfg.state_dim,
            action_dim=cfg.action_dim,
            reward_dim=cfg.reward_dim,
            log_std_min=cfg.log_std_min,
            log_std_max=cfg.log_std_max,
            actor_net=cfg.actor_net,
            critic_net=cfg.critic_net,
            lr=cfg.lr,
            tau=cfg.tau,
            target_entropy=cfg.target_entropy,
            weight_decay=cfg.weight_decay,
            policy_update_freq=cfg.policy_update_freq,
            reward_weight=cfg.reward_weight,
            device=cfg.device,
            use_jit=cfg.use_jit,
            train=cfg.train,
            name=cfg.name,
            load_ckpt=cfg.load_ckpt,
            ckpt_file=cfg.ckpt_file,
        )

    def to(self, device: torch.device, non_blocking: bool = True):
        self.actor.to(device, non_blocking=non_blocking)
        self.Q1.to(device, non_blocking=non_blocking)
        self.Q2.to(device, non_blocking=non_blocking)
        self.Q1_target.to(device, non_blocking=non_blocking)
        self.Q2_target.to(device, non_blocking=non_blocking)

    def train(self, mode: bool = True) -> None:
        self.actor.train(mode)
        self.Q1.train(mode)
        self.Q2.train(mode)

        super().train(mode)

        # unnecessary but just to be sure
        self.Q1_target.eval()
        self.Q2_target.eval()

    def write_logs(
        self,
        metrics: dict[str, Any],
        step: int,
        tb_writer: TBWriter,
        mlflow_writer: MlflowWriter,
    ) -> None:

        mlflow_metrics = {}

        # alpha
        alpha_m = metrics["alpha"]

        tb_writer.add_scalar("train/alpha/loss", alpha_m["alpha_loss"], step)
        tb_writer.add_scalar("train/alpha/value", alpha_m["alpha"], step)
        tb_writer.add_scalar("train/alpha/entropy", alpha_m["sample_entropy"], step)

        mlflow_metrics["alpha_loss"] = alpha_m["alpha_loss"]
        mlflow_metrics["alpha_value"] = alpha_m["alpha"]

        # critic
        critic_m = metrics["critic"]

        tb_writer.add_scalar("train/critic/loss", critic_m["critic_loss"], step)

        tb_writer.add_histogram("train/critic/q1_pred", critic_m["q1_pred"], step)
        tb_writer.add_histogram("train/critic/q2_pred", critic_m["q2_pred"], step)
        tb_writer.add_histogram("train/critic/q_target", critic_m["q_target"], step)

        mlflow_metrics["critic_loss"] = critic_m["critic_loss"]

        # actor
        if "actor" in metrics:
            actor_m = metrics["actor"]

            tb_writer.add_scalar("train/actor/loss", actor_m["actor_loss"], step)

            tb_writer.add_histogram("train/actor/log_prob", actor_m["log_prob"], step)
            tb_writer.add_histogram("train/actor/q", actor_m["q"], step)

            mlflow_metrics["actor_loss"] = actor_m["actor_loss"]

        mlflow_writer.log_metrics(mlflow_metrics, step=step)

from typing import Optional, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import mlflow

from .base import BaseRL
from .transition import TransitionBatch

# https://github.com/rail-berkeley/softlearning/blob/master/softlearning/algorithms/sac.py
# https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/sac/sac.py

# TODO - add lr scheduler
# TODO - support shared base network between actor and critics


def default_target_entropy(action_space: tuple[int, ...]) -> int:
    """
    Compute the default target entropy for a given action space.
    The default target entropy is defined as the negative product
    of all dimensions in the action space.

    :param action_space: Action space
    :type action_space: tuple[int, ...]
    :return: Default target entropy
    :rtype: int
    """
    from functools import reduce
    import operator

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
        load_chkpt: bool = False,
        chkpt_file: Optional[Path] = None,
        **kwargs,
    ):
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_dim = reward_dim

        self.lr = lr
        self.tau = tau
        self.weight_decay = weight_decay
        self.policy_update_freq = policy_update_freq
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.reward_weight = reward_weight.to(self.device)
        self.use_jit = use_jit
        self.train(train)
        self.name = name

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
        # move networks to device
        self.actor.to(self.device)
        self.Q1.to(self.device)
        self.Q2.to(self.device)
        self.Q1_target.to(self.device)
        self.Q2_target.to(self.device)
        self._hard_update()

        # Network Optimizers
        self.actor_optim = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.Q_optim = torch.optim.Adam(
            list(self.Q1.parameters()) + list(self.Q2.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.total_steps = 0

        # load checkpoint before jit compile
        if load_chkpt and chkpt_file:
            self.load(chkpt_file)
        if self.use_jit:
            self._jit_compile()

    def _jit_compile(self):
        """
        Compile model submodules using JIT.

        This method prepares and optimizes sub-components (e.g., actor,
        critic networks) for faster inference or training.

        :return: None
        :rtype: None
        """
        # sampling used in actor is not allowed on jit compilation
        # self.actor = torch.jit.script(self.actor)
        self.Q1 = torch.jit.script(self.Q1)
        self.Q2 = torch.jit.script(self.Q2)

    def get_alpha(self):
        return self.log_alpha.exp().item()

    @torch.no_grad()
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute actions from the current state without gradient tracking.

        :param state: Observation batch. Shape: (N, *obs_shape)
        :type state: torch.Tensor
        :return: Actions. Shape: (N, *action_space)
        :rtype: torch.Tensor
        """
        action, _ = self.actor(state, deterministic=not self._train)

        return action

    @torch.no_grad()
    def _hard_update(self) -> None:
        """
        Hard-update target network parameters:
            θ_target ← θ_source
        """
        # copy weights
        self.Q1_target.load_state_dict(self.Q1.state_dict())
        self.Q2_target.load_state_dict(self.Q2.state_dict())
        # set eval mode
        self.Q1_target.eval()
        self.Q2_target.eval()

    @torch.no_grad()
    def _soft_update(self) -> None:
        """
        Soft-update target network parameters:
            θ_target ← τ * θ_source + (1 - τ) * θ_target
        """
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
        """
        Perform a full SAC update step.

        This method updates alpha, critic, and (periodically) the actor
        using the provided transition batch.
        This method does **not** perform any sampling from the replay buffer;
        it expects the caller to provide the batch (e.g., sampled externally).

        :param transition: Transition batch for update.
        :type transition: TransitionBatch
        :return: Dictionary of scalar metrics aggregated from all update steps:
                 - metrics from :meth:`_update_alpha`
                 - metrics from :meth:`_update_critic`
                 - metrics from :meth:`_update_actor` (if the actor was updated at this step)
        :rtype: Dict[str, float]
        """
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
        """
        Update the temperature parameter (alpha).

        :param transition: Transition batch for update.
        :type transition: TransitionBatch
        :return: Dictionary of scalar metrics including:
                 - ``alpha``: current temperature value.
                 - ``sample_entropy``: estimated policy entropy
                   (negative mean log-probability of sampled actions).
                 - ``alpha_loss``: loss for the alpha update.
        :rtype: Dict[str, float]
        """
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
        """
        Update the critic (Q-function) networks.

        :param transition: Transition batch for update.
        :type transition: TransitionBatch
        :return: Dictionary of scalar metrics including:
                 - ``critic_loss``: loss for the critic update.
        :rtype: Dict[str, float]
        """
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
        """
        Update the policy (actor) network.

        :param transition: Transition batch for update.
        :type transition: TransitionBatch
        :return: Dictionary of scalar metrics including:
                 - ``actor_loss``: loss for the actor update.
        :rtype: Dict[str, float]
        """
        s, a, r, ns, done = transition.unpack()
        alpha = self.log_alpha.detach().exp()

        action, log_prob = self.actor(s)
        q1 = self.Q1(s, action)
        q2 = self.Q2(s, action)
        # q = torch.min(q1, q2) # softlearning
        q = 0.5 * (q1 + q2)  # SB3
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

    def save(self, chkpt_file):
        """
        Save model state to a checkpoint.

        :param chkpt_file: Destination path.
        :type chkpt_file: pathlib.Path
        :return: None
        :rtype: None
        """
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "Q1": self.Q1.state_dict(),
                "Q2": self.Q2.state_dict(),
                "log_alpha": self.log_alpha.detach().cpu(),
                "total_steps": self.total_steps,
            },
            chkpt_file,
        )

    def load(self, chkpt_file):
        """
        Load model state from a checkpoint.

        :param chkpt_file: Source path.
        :type chkpt_file: pathlib.Path
        :return: None
        :rtype: None
        """
        chkpt = torch.load(chkpt_file)
        self.actor.load_state_dict(chkpt["actor"])
        self.Q1.load_state_dict(chkpt["Q1"])
        self.Q2.load_state_dict(chkpt["Q2"])
        # required because log_alpha is saved as a tensor
        self.log_alpha.data.copy_(chkpt["log_alpha"].to(self.device))
        self.total_steps = chkpt["total_steps"]

    def log_params(self, *, prefix: str = "agent/") -> None:
        """
        Log agent hyperparameters to the active MLflow run.

        This method records the agent's key configuration values
        (e.g., discount factor, network dimensions, optimizer settings)
        as parameters in the current MLflow run.
        Note that it **does not** log any checkpoint file that may
        have been loaded at initialization; such artifacts should be
        logged at a higher level (e.g., in the training script).

        :param prefix: String prefix added to all parameter names
                       to group them in MLflow (e.g., ``"agent/"``).
        :type prefix: str
        :raises RuntimeError: If no active MLflow run is found.
        :return: None
        :rtype: None
        """
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
                f"{p}log_std_min": self.log_std_min,
                f"{p}log_std_max": self.log_std_max,
                f"{p}lr": self.lr,
                f"{p}tau": self.tau,
                f"{p}target_entropy": self.target_entropy,
                f"{p}weight_decay": self.weight_decay,
                f"{p}policy_update_freq": self.policy_update_freq,
                f"{p}reward_weight": self.reward_weight.detach().cpu().tolist(),
                f"{p}device": self.device,
                f"{p}use_jit": self._jit_compile,
            }
        )

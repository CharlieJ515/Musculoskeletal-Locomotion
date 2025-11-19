from __future__ import annotations
from dataclasses import dataclass
import warnings
from typing import Union

import torch
import mlflow

from rl.replay_buffer.base import BaseReplayBuffer
from rl.transition import Transition, TransitionBatch


@dataclass
class ReplayBufferConfig:
    capacity: int
    obs_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    reward_shape: tuple[int, ...] = (1,)
    device: torch.device = torch.device("cpu")
    obs_dtype: torch.dtype = torch.float32
    action_dtype: torch.dtype = torch.float32
    reward_dtype: torch.dtype = torch.float32
    name: str = "ReplayBuffer"


class ReplayBuffer(BaseReplayBuffer):
    """
    Plain uniform replay buffer with ring-buffer storage and uniform sampling.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        action_shape: tuple[int, ...],
        reward_shape: tuple[int, ...] = (1,),  # scalar reward => (1,)
        *,
        device: torch.device = torch.device("cpu"),
        obs_dtype: torch.dtype = torch.float32,
        action_dtype: torch.dtype = torch.float32,
        reward_dtype: torch.dtype = torch.float32,
        name: str = "ReplayBuffer",
    ) -> None:
        """
        Initialize the replay buffer with preallocated storage.

        :param capacity: Maximum number of transitions the buffer can hold.
        :param obs_shape: Shape of a single observation.
        :param action_shape: Shape of a single action.
        :param reward_shape: Shape of a single reward (default: scalar (1,)).
        :param obs_dtype: Observations data type.
        :param action_dtype: Actions data type.
        :param reward_dtype: Rewards data type.
        :param device: Device to store data.
        """
        self.capacity = int(capacity)
        self._device = device
        self.name = name

        # storage
        self._obs = torch.empty(
            (capacity, *obs_shape), dtype=obs_dtype, device=self._device
        )
        self._actions = torch.empty(
            (capacity, *action_shape), dtype=action_dtype, device=self._device
        )
        self._rewards = torch.empty(
            (capacity, *reward_shape), dtype=reward_dtype, device=self._device
        )
        self._next_obs = torch.empty(
            (capacity, *obs_shape), dtype=obs_dtype, device=self._device
        )
        self._dones = torch.empty((capacity,), dtype=torch.bool, device=self._device)

        # ring buffer pointers
        self._ptr = 0
        self._size = 0

        # cached shapes
        self._obs_shape = obs_shape
        self._action_shape = action_shape
        self._reward_shape = reward_shape

    # Accept both Transition and TransitionBatch
    def add(self, transition: Union[Transition, TransitionBatch]) -> None:
        """
        Add one or more transitions to the buffer.

        :param transition: Either a single Transition or a TransitionBatch.
                           Shapes:
                             If Transition:
                               - obs: (*obs_shape,)
                               - action: (*action_shape,)
                               - reward: (*reward_shape)
                               - next_obs: (*obs_shape,)
                               - done: ()
                             If TransitionBatch:
                               - obs: (B, *obs_shape)
                               - actions: (B, *action_shape)
                               - rewards: (B, *reward_shape)
                               - next_obs: (B, *obs_shape)
                               - dones: (B,)
        :type transition: Union[Transition, TransitionBatch]
        """
        # normalize to batch
        batch = (
            transition.to_batch() if isinstance(transition, Transition) else transition
        )

        B = len(batch)
        if B <= 0:
            return

        # sanity check on shapes
        if tuple(batch.obs.shape[1:]) != self._obs_shape:
            raise ValueError(
                f"obs shape mismatch: got {tuple(batch.obs.shape[1:])}, expected {self._obs_shape}"
            )
        if tuple(batch.actions.shape[1:]) != self._action_shape:
            raise ValueError(
                f"actions shape mismatch: got {tuple(batch.actions.shape[1:])}, expected {self._action_shape}"
            )
        if tuple(batch.rewards.shape[1:]) != self._reward_shape:
            raise ValueError(
                f"rewards shape mismatch: got {tuple(batch.rewards.shape[1:])}, expected {self._reward_shape}"
            )

        # move to buffer device
        batch = batch.to(self._device, non_blocking=True)

        # ring buffer write (handle wrap-around)
        end = self._ptr + B
        if end <= self.capacity:
            sl = slice(self._ptr, end)
            self._obs[sl] = batch.obs
            self._actions[sl] = batch.actions
            self._rewards[sl] = batch.rewards
            self._next_obs[sl] = batch.next_obs
            self._dones[sl] = batch.dones
        else:
            first = self.capacity - self._ptr
            second = B - first
            # first chunk
            sl1 = slice(self._ptr, self.capacity)
            self._obs[sl1] = batch.obs[:first]
            self._actions[sl1] = batch.actions[:first]
            self._rewards[sl1] = batch.rewards[:first]
            self._next_obs[sl1] = batch.next_obs[:first]
            self._dones[sl1] = batch.dones[:first]
            # wrap chunk
            sl2 = slice(0, second)
            self._obs[sl2] = batch.obs[first:]
            self._actions[sl2] = batch.actions[first:]
            self._rewards[sl2] = batch.rewards[first:]
            self._next_obs[sl2] = batch.next_obs[first:]
            self._dones[sl2] = batch.dones[first:]

        self._ptr = (self._ptr + B) % self.capacity
        self._size = min(self._size + B, self.capacity)

    def sample(self, batch_size: int, *, pin_memory: bool = True) -> TransitionBatch:
        """
        Sample a batch of transitions from the buffer.

        :param batch_size: Number of transitions to sample.
        :type batch_size: int
        :return: A batch of transitions with tensors already stacked.
                 Shapes:
                   - obs: (B, *obs_shape)
                   - actions: (B, *action_shape)
                   - rewards: (B, *reward_shape)
                   - next_obs: (B, *obs_shape)
                   - dones: (B,)
        :rtype: TransitionBatch
        """
        if self._size == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        batch_size = min(batch_size, self._size)
        idx = torch.randint(0, self._size, (batch_size,), device=self._device)

        if pin_memory and self._device.type == "cuda":
            warnings.warn(
                "pin_memory=True has no effect because the buffer is already on CUDA.",
                stacklevel=2,
            )

        if pin_memory and self._device.type == "cpu":
            obs = torch.empty(
                (batch_size, *self._obs_shape), dtype=self._obs.dtype, pin_memory=True
            )
            actions = torch.empty(
                (batch_size, *self._action_shape),
                dtype=self._actions.dtype,
                pin_memory=True,
            )
            rewards = torch.empty(
                (batch_size, *self._reward_shape),
                dtype=self._rewards.dtype,
                pin_memory=True,
            )
            next_obs = torch.empty(
                (batch_size, *self._obs_shape),
                dtype=self._next_obs.dtype,
                pin_memory=True,
            )
            dones = torch.empty((batch_size,), dtype=self._dones.dtype, pin_memory=True)

            torch.index_select(self._obs, 0, idx, out=obs)
            torch.index_select(self._actions, 0, idx, out=actions)
            torch.index_select(self._rewards, 0, idx, out=rewards)
            torch.index_select(self._next_obs, 0, idx, out=next_obs)
            torch.index_select(self._dones, 0, idx, out=dones)
        else:
            obs = self._obs.index_select(0, idx)
            actions = self._actions.index_select(0, idx)
            rewards = self._rewards.index_select(0, idx)
            next_obs = self._next_obs.index_select(0, idx)
            dones = self._dones.index_select(0, idx)

        new = TransitionBatch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
        )
        return new

    def __len__(self) -> int:
        """
        :return: Current number of stored transitions.
        :rtype: int
        """
        return self._size

    def clear(self) -> None:
        """
        Reset buffer (remove all stored transitions).
        """
        self._ptr = 0
        self._size = 0

    def device(self) -> torch.device:
        """
        :return: Device on which the buffer’s data is stored.
        :rtype: torch.device
        """
        return self._device

    def log_params(self, *, prefix: str = "buffer/") -> None:
        """
        Log replay buffer configuration to the active MLflow run.

        This method records key configuration parameters of the replay buffer
        (e.g., capacity, observation/action/reward shapes, and device)
        as parameters in the current MLflow run.

        :param prefix: String prefix added to all parameter names
                       to group them in MLflow (e.g., ``"buffer/"``).
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
                f"{p}capacity": self.capacity,
                f"{p}obs_shape": self._obs_shape,
                f"{p}action_shape": self._action_shape,
                f"{p}reward_shape": self._reward_shape,
                f"{p}device": self.device,
            }
        )

    def all(self, *, pin_memory: bool = True) -> TransitionBatch:
        """
        Return all currently stored transitions as a single TransitionBatch.

        :param pin_memory: Whether to pin memory for faster GPU transfer (only effective on CPU).
        :type pin_memory: bool
        :return: TransitionBatch containing all stored transitions.
        :rtype: TransitionBatch
        """
        if self._size == 0:
            raise ValueError("Replay buffer is empty. Nothing to return.")

        if pin_memory and self._device.type == "cuda":
            warnings.warn(
                "pin_memory=True has no effect because the buffer is already on CUDA.",
                stacklevel=2,
            )

        if pin_memory and self._device.type == "cpu":
            obs = self._obs[: self._size].clone().pin_memory()
            actions = self._actions[: self._size].clone().pin_memory()
            rewards = self._rewards[: self._size].clone().pin_memory()
            next_obs = self._next_obs[: self._size].clone().pin_memory()
            dones = self._dones[: self._size].clone().pin_memory()
        else:
            obs = self._obs[: self._size].clone()
            actions = self._actions[: self._size].clone()
            rewards = self._rewards[: self._size].clone()
            next_obs = self._next_obs[: self._size].clone()
            dones = self._dones[: self._size].clone()

        return TransitionBatch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
        )

    @classmethod
    def from_config(cls, cfg: "ReplayBufferConfig") -> "ReplayBuffer":
        return cls(
            capacity=cfg.capacity,
            obs_shape=cfg.obs_shape,
            action_shape=cfg.action_shape,
            reward_shape=cfg.reward_shape,
            device=cfg.device,
            obs_dtype=cfg.obs_dtype,
            action_dtype=cfg.action_dtype,
            reward_dtype=cfg.reward_dtype,
            name=cfg.name,
        )

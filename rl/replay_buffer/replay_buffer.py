from __future__ import annotations

import warnings
from typing import Union

import numpy as np
import torch

from analysis import MlflowWriter
from configs import ReplayBufferConfig
from rl.replay_buffer.base import BaseReplayBuffer
from rl.transition import Transition, TransitionBatch


class ReplayBuffer(BaseReplayBuffer):
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
        self.capacity = int(capacity)
        self._device = device
        self.name = name

        # storage
        self._obs = torch.zeros(
            (capacity, *obs_shape),
            dtype=obs_dtype,
            device=self._device,
        )
        self._actions = torch.zeros(
            (capacity, *action_shape),
            dtype=action_dtype,
            device=self._device,
        )
        self._rewards = torch.zeros(
            (capacity, *reward_shape),
            dtype=reward_dtype,
            device=self._device,
        )
        self._next_obs = torch.zeros(
            (capacity, *obs_shape),
            dtype=obs_dtype,
            device=self._device,
        )
        self._dones = torch.zeros((capacity,), dtype=torch.bool, device=self._device)

        # ring buffer pointers
        self._ptr = 0
        self._size = 0

        # shapes
        self._obs_shape = obs_shape
        self._action_shape = action_shape
        self._reward_shape = reward_shape

    def add(self, transition: Union[Transition, TransitionBatch]) -> None:
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
        if self._size == 0:
            raise ValueError("Cannot sample from an empty buffer.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        batch_size = min(batch_size, self._size)
        idx = torch.randint(
            0, self._size, (batch_size,), dtype=torch.long, device=self._device
        )

        batch = self.sample_index(idx, pin_memory=pin_memory)
        return batch

    def sample_index(
        self,
        index: torch.Tensor | np.ndarray,
        *,
        pin_memory: bool = True,
    ) -> TransitionBatch:
        idx = torch.as_tensor(index, dtype=torch.long, device=self._device)
        if idx.max() >= self._size:
            raise ValueError()

        if pin_memory and self._device.type == "cuda":
            warnings.warn(
                "pin_memory=True has no effect because the buffer is already on CUDA.",
                stacklevel=2,
            )

        if pin_memory and self._device.type == "cpu":
            batch_size = len(index)
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

        batch = TransitionBatch(
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
            indices=idx,
        )
        return batch

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        self._ptr = 0
        self._size = 0

    def device(self) -> torch.device:
        return self._device

    def log_params(
        self, mlflow_writer: MlflowWriter, *, prefix: str = "buffer/"
    ) -> None:
        p = prefix

        params_dict = {
            f"{p}name": self.name,
            f"{p}capacity": self.capacity,
            f"{p}obs_shape": self._obs_shape,
            f"{p}action_shape": self._action_shape,
            f"{p}reward_shape": self._reward_shape,
            f"{p}device": str(self._device),
        }

        mlflow_writer.log_params(params_dict)

    def all(self, *, pin_memory: bool = True) -> TransitionBatch:
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
    def from_config(cls, cfg: ReplayBufferConfig) -> "ReplayBuffer":
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

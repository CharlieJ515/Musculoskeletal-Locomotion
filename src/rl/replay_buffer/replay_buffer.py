from __future__ import annotations

import warnings
from typing import Tuple, Union
import torch

from rl.replay_buffer.base import BaseReplayBuffer
from utils.transition import Transition, TransitionBatch


class ReplayBuffer(BaseReplayBuffer):
    """
    Plain uniform replay buffer with ring-buffer storage and uniform sampling.
    Stores transitions as preallocated tensors for speed.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...],
        act_shape: Tuple[int, ...],
        reward_shape: Tuple[int, ...] = (1,),  # scalar reward => (1,)
        *,
        device: torch.device = torch.device("cpu"),
        obs_dtype: torch.dtype = torch.float32,
        act_dtype: torch.dtype = torch.float32,
        rew_dtype: torch.dtype = torch.float32,
    ) -> None:
        """
        Initialize the replay buffer with preallocated storage.

        :param capacity: Maximum number of transitions the buffer can hold.
        :param obs_shape: Shape of a single observation.
        :param act_shape: Shape of a single action.
        :param reward_shape: Shape of a single reward (default: scalar (1,)).
        :param obs_dtype: Data type for storing observations.
        :param act_dtype: Data type for storing actions.
        :param rew_dtype: Data type for storing rewards.
        :param device: Device to store data on (CPU or GPU).
        """
        self.capacity = int(capacity)
        self._device = device

        # storage
        self._obs = torch.empty((capacity, *obs_shape), dtype=obs_dtype, device=self._device)
        self._actions = torch.empty((capacity, *act_shape), dtype=act_dtype, device=self._device)
        self._rewards = torch.empty((capacity, *reward_shape), dtype=rew_dtype, device=self._device)
        self._next_obs = torch.empty((capacity, *obs_shape), dtype=obs_dtype, device=self._device)
        self._dones = torch.empty((capacity,), dtype=torch.bool, device=self._device)

        # ring buffer pointers
        self._ptr = 0
        self._size = 0

        # cached shapes
        self._obs_shape = obs_shape
        self._act_shape = act_shape
        self._rew_shape = reward_shape

    # Accept both Transition and TransitionBatch
    def add(self, transition: Union[Transition, TransitionBatch]) -> None:
        """
        Add one or more transitions to the buffer.

        :param transition: Either a single Transition or a TransitionBatch.
                           Shapes:
                             If Transition:
                               - obs: (*obs_shape,)
                               - action: (*act_shape,)
                               - reward: (*reward_shape)
                               - next_obs: (*obs_shape,)
                               - done: ()
                             If TransitionBatch:
                               - obs: (B, *obs_shape)
                               - actions: (B, *act_shape)
                               - rewards: (B, *reward_shape)
                               - next_obs: (B, *obs_shape)
                               - dones: (B,)
        :type transition: Union[Transition, TransitionBatch]
        """

        # normalize to batch
        batch = transition.to_batch() if isinstance(transition, Transition) else transition

        B = len(batch)
        if B <= 0:
            return

        # sanity check on shapes
        if tuple(batch.obs.shape[1:]) != self._obs_shape:
            raise ValueError(f"obs shape mismatch: got {tuple(batch.obs.shape[1:])}, expected {self._obs_shape}")
        if tuple(batch.actions.shape[1:]) != self._act_shape:
            raise ValueError(f"actions shape mismatch: got {tuple(batch.actions.shape[1:])}, expected {self._act_shape}")
        if tuple(batch.rewards.shape[1:]) != self._rew_shape:
            raise ValueError(f"rewards shape mismatch: got {tuple(batch.rewards.shape[1:])}, expected {self._rew_shape}")

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

    def sample(self, batch_size: int, *, pin_memory:bool=True) -> TransitionBatch:
        """
        Sample a batch of transitions from the buffer.

        :param batch_size: Number of transitions to sample.
        :type batch_size: int
        :param pin_memory: If True and storage is on CPU, gather directly into pinned
                           tensors (better for async .to('cuda')).
        :type pin_memory: bool
        :return: A batch of transitions with tensors already stacked.
                 Shapes:
                   - obs: (B, *obs_shape)
                   - actions: (B, *act_shape)
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
                stacklevel=2
            )

        if pin_memory and self._device.type == "cpu":
            obs     = torch.empty((batch_size, *self._obs_shape), dtype=self._obs.dtype,     pin_memory=True)
            actions = torch.empty((batch_size, *self._act_shape), dtype=self._actions.dtype, pin_memory=True)
            rewards = torch.empty((batch_size, *self._rew_shape), dtype=self._rewards.dtype, pin_memory=True)
            next_obs= torch.empty((batch_size, *self._obs_shape), dtype=self._next_obs.dtype,pin_memory=True)
            dones   = torch.empty((batch_size,),                  dtype=self._dones.dtype,   pin_memory=True)

            torch.index_select(self._obs,      0, idx, out=obs)
            torch.index_select(self._actions,  0, idx, out=actions)
            torch.index_select(self._rewards,  0, idx, out=rewards)
            torch.index_select(self._next_obs, 0, idx, out=next_obs)
            torch.index_select(self._dones,    0, idx, out=dones)
        else:
            obs      = self._obs.index_select(0, idx)
            actions  = self._actions.index_select(0, idx)
            rewards  = self._rewards.index_select(0, idx)
            next_obs = self._next_obs.index_select(0, idx)
            dones    = self._dones.index_select(0, idx)

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
        Reset the buffer (remove all stored transitions).
        """
        self._ptr = 0
        self._size = 0

    def device(self) -> torch.device:
        """
        :return: Device on which the buffer’s data is stored.
        :rtype: torch.device
        """
        return self._device


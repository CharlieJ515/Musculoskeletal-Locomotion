from abc import ABC, abstractmethod
from typing import Union

import torch

from analysis import MlflowWriter
from rl.transition import Transition, TransitionBatch


class BaseReplayBuffer(ABC):
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
        self.name = name
        self._device = device

        self._obs_shape = obs_shape
        self._action_shape = action_shape
        self._reward_shape = reward_shape

        self._obs = torch.zeros(
            (self.capacity, *self._obs_shape),
            dtype=obs_dtype,
            device=self._device,
        )
        self._actions = torch.zeros(
            (self.capacity, *self._action_shape),
            dtype=action_dtype,
            device=self._device,
        )
        self._rewards = torch.zeros(
            (self.capacity, *self._reward_shape),
            dtype=reward_dtype,
            device=self._device,
        )
        self._next_obs = torch.zeros(
            (self.capacity, *self._obs_shape),
            dtype=obs_dtype,
            device=self._device,
        )
        self._dones = torch.zeros(
            (self.capacity,), dtype=torch.bool, device=self._device
        )

        self._ptr = 0
        self._size = 0

    @abstractmethod
    def add(
        self,
        transition: Union[Transition, TransitionBatch],
    ) -> None: ...

    @abstractmethod
    def sample(
        self, batch_size: int, *, pin_memory: bool = True
    ) -> TransitionBatch: ...

    def __len__(self) -> int:
        return self._size

    def clear(self) -> None:
        self._ptr = 0
        self._size = 0

    @property
    def device(self) -> torch.device:
        return self._device

    @abstractmethod
    def log_params(
        self, mlflow_writer: MlflowWriter, *, prefix: str = "buffer/"
    ) -> None: ...

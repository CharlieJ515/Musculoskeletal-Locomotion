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
        act_dtype: torch.dtype = torch.float32,
        rew_dtype: torch.dtype = torch.float32,
    ) -> None: ...

    @abstractmethod
    def add(
        self,
        transition: Union[Transition, TransitionBatch],
    ) -> None: ...

    @abstractmethod
    def sample(self, batch_size: int) -> TransitionBatch: ...

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def clear(self) -> None: ...

    @abstractmethod
    def device(self) -> torch.device: ...

    @abstractmethod
    def log_params(
        self, mlflow_writer: MlflowWriter, *, prefix: str = "buffer/"
    ) -> None: ...

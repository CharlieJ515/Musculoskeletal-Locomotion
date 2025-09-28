from abc import ABC, abstractmethod
from typing import Union
import torch
from src.types import Transition, TransitionBatch

class BaseReplayBuffer(ABC):
    """
    Abstract interface for a replay buffer.
    """

    @abstractmethod
    def add(
        self,
        transition: Union[Transition, TransitionBatch],
    ) -> None:
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

    @abstractmethod
    def sample(self, batch_size: int) -> TransitionBatch:
        """
        Sample a batch of transitions from the buffer.

        :param batch_size: Number of transitions to sample.
        :type batch_size: int
        :return: A batch of transitions with tensors already stacked.
                 Shapes:
                   - obs: (B, *obs_shape)
                   - actions: (B, *act_shape)
                   - rewards: (B, *reward_shape)
                   - next_obs: (B, *obs_shape)
                   - dones: (B,)
        :rtype: TransitionBatch
        """

    @abstractmethod
    def __len__(self) -> int:
        """
        :return: Current number of stored transitions.
        :rtype: int
        """

    @abstractmethod
    def clear(self) -> None:
        """
        Reset the buffer (remove all stored transitions).
        """

    @abstractmethod
    def device(self) -> torch.device:
        """
        :return: Device on which the buffer’s data is stored.
        :rtype: torch.device
        """

    @abstractmethod
    def to(self, device: torch.device) -> "BaseReplayBuffer":
        """
        Move the buffer to a given device.

        :param device: Target device (e.g., ``torch.device('cuda')``).
        :type device: torch.device
        :return: Self, with data moved to the specified device.
        :rtype: BaseReplayBuffer
        """

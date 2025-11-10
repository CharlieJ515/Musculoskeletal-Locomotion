from abc import ABC, abstractmethod
from typing import Union
import torch
from rl.transition import Transition, TransitionBatch


class BaseReplayBuffer(ABC):
    """
    Abstract interface for a replay buffer.
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
        act_dtype: torch.dtype = torch.float32,
        rew_dtype: torch.dtype = torch.float32,
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

    @abstractmethod
    def sample(self, batch_size: int) -> TransitionBatch:
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

    @abstractmethod
    def __len__(self) -> int:
        """
        :return: Current number of stored transitions.
        :rtype: int
        """

    @abstractmethod
    def clear(self) -> None:
        """
        Reset buffer (remove all stored transitions).
        """

    @abstractmethod
    def device(self) -> torch.device:
        """
        :return: Device on which the buffer’s data is stored.
        :rtype: torch.device
        """

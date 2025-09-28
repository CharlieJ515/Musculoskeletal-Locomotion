from dataclasses import dataclass
from typing import Optional
import torch

from . import StreamObject

@dataclass
class Transition(StreamObject):
    """
    A single transition stored in the replay buffer.

    :param obs: Observation at time t. Shape: (*obs_shape,)
    :type obs: torch.Tensor
    :param action: Action taken at time t. Shape: (*act_shape,)
    :type action: torch.Tensor
    :param reward: Reward received after taking the action. Shape: (*reward_shape)
    :type reward: torch.Tensor
    :param next_obs: Observation at time t+1. Shape: (*obs_shape,)
    :type next_obs: torch.Tensor
    :param done: Whether the episode terminated after this step. Shape: ()
    :type done: torch.Tensor
    """
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_obs: torch.Tensor
    done: torch.Tensor

    def __post_init__(self):
        # check device
        dev = self.obs.device
        for name, tensor in [
            ("action", self.action),
            ("reward", self.reward),
            ("next_obs", self.next_obs),
            ("done", self.done),
        ]:
            if tensor.device != dev:
                raise ValueError(
                    f"Device mismatch: field '{name}' is on {tensor.device}, "
                    f"but 'obs' is on {dev}."
                )

        if self.next_obs.shape != self.obs.shape:
            raise ValueError(
                f"'next_obs' shape (excluding batch) {tuple(self.next_obs.shape)} "
                f"must match 'obs' shape {tuple(self.obs.shape)}"
            )
        if self.reward.dim() < 1:
            raise ValueError(f"'reward' must be at least 1-D (e.g., shape (1,)), got {tuple(self.reward.shape)}")
        if self.done.dim() != 0:
            raise ValueError(f"'done' must be a scalar tensor with shape (), got {tuple(self.done.shape)}")

    @property
    def device(self) -> torch.device:
        """
        :return: Device where this transition’s tensors live.
        :rtype: torch.device
        """
        return self.obs.device

    @property
    def observation_space(self) -> torch.Size:
        """
        Shape of the observation.
        """
        return self.obs.shape

    @property
    def action_space(self) -> torch.Size:
        """
        Shape of the action.
        """
        return self.action.shape

    @property
    def reward_shape(self) -> torch.Size:
        """
        Shape of the reward.
        """
        return self.reward.shape

    def to(self, device: torch.device, non_blocking: bool = False) -> "Transition":
        """
        Move all tensors to the given device and return a new Transition.

        :param device: Target device.
        :type device: torch.device
        :param non_blocking: Perform copy asynchronously if possible.
        :type non_blocking: bool

        :return: A new Transition on the target device.
        :rtype: Transition
        """
        if device == self.device:
            return self

        stream_device = device if device.type=="cuda" else self.device
        with self.stream(stream_device):
            new = Transition(
                obs=self.obs.to(device, non_blocking=non_blocking),
                action=self.action.to(device, non_blocking=non_blocking),
                reward=self.reward.to(device, non_blocking=non_blocking),
                next_obs=self.next_obs.to(device, non_blocking=non_blocking),
                done=self.done.to(device, non_blocking=non_blocking),
            )
        new._stream = self._stream
        new._event = self._event
        return new
            
    def to_batch(self) -> "TransitionBatch":
        """
        Convert this single transition into a TransitionBatch of size 1.

        Note:
            We do NOT call `wait()` here. `unsqueeze` only creates a view of the
            original tensor, so no GPU work is launched.  Synchronizing here would
            force the current (often default) stream to wait, which could serialize
            unrelated work and reduce overlap.

            Instead we propagate the producer's event to the new batch so that
            downstream consumers can wait on it in their chosen stream if needed.

        :return: TransitionBatch with leading dimension 1.
        :rtype: TransitionBatch
        """
        batch = TransitionBatch(
            obs=self.obs.unsqueeze(0),
            actions=self.action.unsqueeze(0),
            rewards=self.reward.unsqueeze(0),
            next_obs=self.next_obs.unsqueeze(0),
            dones=self.done.unsqueeze(0),
        )
        batch._stream = self._stream
        batch._event = self._event
        return batch

@dataclass
class TransitionBatch(StreamObject):
    """
    A batch of transitions sampled from the replay buffer.

    :param obs: Observations at time t. Shape: (B, *obs_shape)
    :type obs: torch.Tensor
    :param actions: Actions taken at time t. Shape: (B, *act_shape)
    :type actions: torch.Tensor
    :param rewards: Rewards received after taking the actions. Shape: (B, *reward_shape)
    :type rewards: torch.Tensor
    :param next_obs: Next observations at time t+1. Shape: (B, *obs_shape)
    :type next_obs: torch.Tensor
    :param dones: Episode termination flags. Shape: (B,)
    :type dones: torch.Tensor
    """
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor

    def __post_init__(self):
        """
        Sanity-check that all tensors reside on the same device.
        """
        dev = self.obs.device
        for name, tensor in [
            ("actions", self.actions),
            ("rewards", self.rewards),
            ("next_obs", self.next_obs),
            ("dones", self.dones),
        ]:
            if tensor.device != dev:
                raise ValueError(
                    f"Device mismatch: field '{name}' is on {tensor.device}, "
                    f"but 'obs' is on {dev}."
                )

        B = self.obs.shape[0]
        for name, tensor in [
            ("actions", self.actions),
            ("rewards", self.rewards),
            ("next_obs", self.next_obs),
            ("dones", self.dones),
        ]:
            if tensor.shape[0] != B:
                raise ValueError(f"Leading dim mismatch: '{name}' has B={tensor.shape[0]}, but obs has B={B}")
            
        if self.next_obs.shape[1:] != self.obs.shape[1:]:
            raise ValueError(
                f"'next_obs' shape (excluding batch) {tuple(self.next_obs.shape[1:])} "
                f"must match 'obs' shape {tuple(self.obs.shape[1:])}"
            )
        if self.rewards.dim() < 2:
            raise ValueError(f"'rewards' must be at least 2-D (e.g., (B,1)), got {tuple(self.rewards.shape)}")
        if self.dones.dim() != 1:
            raise ValueError(f"'dones' must be 1-D with shape (B,), got {tuple(self.dones.shape)}")


    def __len__(self) -> int:
        """
        :return: Batch length B (number of transitions).
        :rtype: int
        """
        return int(self.obs.shape[0])

    @property
    def device(self) -> torch.device:
        """
        :return: Device where the batch tensors live.
        :rtype: torch.device
        """
        return self.obs.device

    @property
    def observation_space(self) -> torch.Size:
        """
        Shape of a single observation (excluding batch dimension).
        """
        return self.obs.shape[1:]

    @property
    def action_space(self) -> torch.Size:
        """
        Shape of a single action (excluding batch dimension).
        """
        return self.actions.shape[1:]

    @property
    def reward_shape(self) -> torch.Size:
        """
        Reward shape excluding the batch dimension.
        """
        return self.rewards.shape[1:]

    def to(self, device: torch.device, non_blocking: bool = False) -> "TransitionBatch":
        """
        Move all tensors to the given device and return a new batch.

        :param device: Target device.
        :type device: torch.device
        :param non_blocking: Perform copy asynchronously if possible.
        :type non_blocking: bool
        :return: A new TransitionBatch on the target device.
        :rtype: TransitionBatch
        """
        if device == self.device:
            return self

        stream_device = device if device.type == "cuda" else self.device
        with self.stream(stream_device):
            new = TransitionBatch(
                obs=self.obs.to(device, non_blocking=non_blocking),
                actions=self.actions.to(device, non_blocking=non_blocking),
                rewards=self.rewards.to(device, non_blocking=non_blocking),
                next_obs=self.next_obs.to(device, non_blocking=non_blocking),
                dones=self.dones.to(device, non_blocking=non_blocking),
            )
        new._stream = self._stream
        new._event = self._event
        return new

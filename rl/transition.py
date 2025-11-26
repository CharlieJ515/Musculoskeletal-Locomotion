from dataclasses import dataclass
import torch


@dataclass(frozen=True, slots=True)
class Transition:
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

    index: torch.Tensor | None = None
    weight: torch.Tensor | None = None

    def __post_init__(self):
        dev = self.obs.device
        name: str
        tensor: torch.Tensor | None
        for name, tensor in [
            ("action", self.action),
            ("reward", self.reward),
            ("next_obs", self.next_obs),
            ("done", self.done),
            ("index", self.index),
            ("weight", self.weight),
        ]:
            if tensor is None:
                continue
            # check device
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

        for name, tensor in [
            ("reward", self.reward),
            ("weight", self.weight),
        ]:
            if tensor is None or tensor.dim() >= 1:
                continue
            raise ValueError(
                f"'reward' must be at least 1-D (e.g., shape (1,)), got {tuple(tensor.shape)}"
            )

        for name, tensor in [
            ("done", self.done),
            ("index", self.index),
        ]:
            if tensor is None or tensor.dim() == 0:
                continue
            raise ValueError(
                f"'{name}' must be a scalar tensor with shape (), got {tuple(tensor.shape)}"
            )

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

        index = (
            self.index.to(device, non_blocking=non_blocking)
            if self.index is not None
            else None
        )
        weight = (
            self.weight.to(device, non_blocking=non_blocking)
            if self.weight is not None
            else None
        )

        new = Transition(
            obs=self.obs.to(device, non_blocking=non_blocking),
            action=self.action.to(device, non_blocking=non_blocking),
            reward=self.reward.to(device, non_blocking=non_blocking),
            next_obs=self.next_obs.to(device, non_blocking=non_blocking),
            done=self.done.to(device, non_blocking=non_blocking),
            index=index,
            weight=weight,
        )
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
        indices = self.index.unsqueeze(0) if self.index is not None else None
        weights = self.weight.unsqueeze(0) if self.weight is not None else None

        batch = TransitionBatch(
            obs=self.obs.unsqueeze(0),
            actions=self.action.unsqueeze(0),
            rewards=self.reward.unsqueeze(0),
            next_obs=self.next_obs.unsqueeze(0),
            dones=self.done.unsqueeze(0),
            indices=indices,
            weights=weights,
        )
        return batch

    def unpack(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unpack the stored transition components.

        This method returns the five core elements of a transition in the order of
        current observation, action, reward, next observation, and done flag.

        :return: A tuple containing:
                 - ``obs`` (:class:`torch.Tensor`): Current observations.
                 - ``action`` (:class:`torch.Tensor`): Actions taken.
                 - ``reward`` (:class:`torch.Tensor`): Rewards received after taking the actions.
                 - ``next_obs`` (:class:`torch.Tensor`): Next observations after the actions.
                 - ``done`` (:class:`torch.Tensor`): Boolean or float flags indicating episode termination.
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        return (self.obs, self.action, self.reward, self.next_obs, self.done)


@dataclass(frozen=True, slots=True)
class TransitionBatch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor

    indices: torch.Tensor | None = None
    weights: torch.Tensor | None = None

    def __post_init__(self):
        # check device
        device = self.obs.device
        B = self.obs.shape[0]
        name: str
        tensor: torch.Tensor | None
        for name, tensor in [
            ("actions", self.actions),
            ("rewards", self.rewards),
            ("next_obs", self.next_obs),
            ("dones", self.dones),
            ("indices", self.indices),
            ("weights", self.weights),
        ]:
            if tensor is None:
                continue
            # check device
            if tensor.device != device:
                raise ValueError(
                    f"Device mismatch: field '{name}' is on {tensor.device}, "
                    f"but 'obs' is on {device}."
                )
            # check batch size
            if tensor.shape[0] != B:
                raise ValueError(
                    f"Leading dim mismatch: '{name}' has B={tensor.shape[0]}, but obs has B={B}"
                )

        if self.next_obs.shape != self.obs.shape:
            raise ValueError(
                f"'next_obs' shape (excluding batch) {tuple(self.next_obs.shape[1:])} "
                f"must match 'obs' shape {tuple(self.obs.shape[1:])}"
            )

        for name, tensor in [
            ("rewards", self.rewards),
            ("weights", self.weights),
        ]:
            if tensor is None or tensor.dim() >= 2:
                continue
            raise ValueError(
                f"'rewards' must be at least 2-D (e.g., (B,1)), got {tuple(tensor.shape)}"
            )

        for name, tensor in [
            ("dones", self.dones),
            ("indices", self.indices),
        ]:
            if tensor is None or tensor.dim() == 1:
                continue
            raise ValueError(
                f"'{name}' must be 1-D with shape (B,), got {tuple(tensor.shape)}"
            )

    def __len__(self) -> int:
        return self.obs.shape[0]

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

        indices = (
            self.indices.to(device, non_blocking=non_blocking)
            if self.indices is not None
            else None
        )
        weights = (
            self.weights.to(device, non_blocking=non_blocking)
            if self.weights is not None
            else None
        )

        new = TransitionBatch(
            obs=self.obs.to(device, non_blocking=non_blocking),
            actions=self.actions.to(device, non_blocking=non_blocking),
            rewards=self.rewards.to(device, non_blocking=non_blocking),
            next_obs=self.next_obs.to(device, non_blocking=non_blocking),
            dones=self.dones.to(device, non_blocking=non_blocking),
            indices=indices,
            weights=weights,
        )
        return new

    def unpack(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Unpack the stored transition components.

        This method returns the five core elements of a transition in the order of
        current observation, action, reward, next observation, and done flag.

        :return: A tuple containing:
                 - ``obs`` (:class:`torch.Tensor`): Current observations.
                 - ``actions`` (:class:`torch.Tensor`): Actions taken.
                 - ``rewards`` (:class:`torch.Tensor`): Rewards received after taking the actions.
                 - ``next_obs`` (:class:`torch.Tensor`): Next observations after the actions.
                 - ``dones`` (:class:`torch.Tensor`): Boolean or float flags indicating episode termination.
        :rtype: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        """
        return (self.obs, self.actions, self.rewards, self.next_obs, self.dones)

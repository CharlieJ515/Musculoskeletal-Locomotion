from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
import torch

from .transition import TransitionBatch


class BaseRL(ABC):
    @abstractmethod
    def _jit_compile(self) -> None:
        """
        Compile model submodules using JIT.

        This method prepares and optimizes sub-components (e.g., actor,
        critic networks) for faster inference or training.

        :return: None
        :rtype: None
        """

    def train(self, mode: bool = True) -> None:
        """
        Set training or evaluation mode.

        :param mode: If ``True``, enables training mode; if ``False``, switches to evaluation mode.
        :type mode: bool
        :return: None
        :rtype: None
        """
        self._train = mode

    def eval(self) -> None:
        """
        Switch to evaluation mode.

        Convenience wrapper for :py:meth:`train(False)`.

        :return: None
        :rtype: None
        """
        self.train(False)

    @abstractmethod
    @torch.no_grad()
    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Compute actions from the current state without gradient tracking.

        :param state: Observation batch. Shape: (N, *obs_shape)
        :type state: torch.Tensor
        :return: Actions. Shape: (N, *action_space)
        :rtype: torch.Tensor
        """

    @abstractmethod
    def update(self, transition: TransitionBatch) -> dict[str, float]:
        """
        Update the model parameters using the given transition.

        This method does **not** perform any sampling from the replay buffer;
        it expects the caller to provide the batch (e.g., sampled externally).

        :param transition: Transition batch provided by the caller.
        :type transition: TransitionBatch
        :return: Scalar metrics such as losses or entropy for logging.
        :rtype: Dict[str, float]
        """

    @abstractmethod
    def save(self, chkpt_file: Path) -> None:
        """
        Save model state to a checkpoint.

        :param chkpt_file: Destination path.
        :type chkpt_file: pathlib.Path
        :return: None
        :rtype: None
        """

    @abstractmethod
    def load(self, chkpt_file: Path) -> None:
        """
        Load model state from a checkpoint.

        :param chkpt_file: Source path.
        :type chkpt_file: pathlib.Path
        :return: None
        :rtype: None
        """

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch

from analysis import MlflowWriter, TBWriter

from .transition import TransitionBatch


class BaseRL(ABC):
    def __init__(self, device: torch.device | None, gamma: float):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.gamma = gamma

    @abstractmethod
    def _jit_compile(self) -> None: ...

    def train(self, mode: bool = True) -> None:
        self._train = mode

    def eval(self) -> None:
        self.train(False)

    @abstractmethod
    @torch.no_grad()
    def select_action(self, state: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def update(self, transition: TransitionBatch) -> dict[str, Any]: ...

    @abstractmethod
    def save(self, ckpt_file: Path) -> None: ...

    @abstractmethod
    def load(self, ckpt_file: Path) -> None: ...

    @abstractmethod
    def to(self, device: torch.device, non_blocking: bool = True) -> None: ...

    @abstractmethod
    def log_params(
        self, mlflow_writer: MlflowWriter, *, prefix: str = "agent/"
    ) -> None: ...

    @abstractmethod
    def write_logs(
        self,
        metrics: dict[str, Any],
        step: int,
        tb_writer: TBWriter,
        mlflow_writer: MlflowWriter,
    ) -> None: ...

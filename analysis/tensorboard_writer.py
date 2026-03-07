import os
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class TBWriter:
    def __init__(self, log_dir: str | None = None):
        self.is_enabled = os.getenv("ENABLE_TENSORBOARD", "true").lower() == "true"
        self.log_dir = log_dir
        self._writer: SummaryWriter | None = None

        self.log_scalars = True
        self.log_histograms = True

    @property
    def writer(self) -> SummaryWriter | None:
        if not self.is_enabled:
            return None

        if self._writer is None:
            self._writer = SummaryWriter(self.log_dir)

        return self._writer

    def set_enabled(self, state: bool) -> None:
        self.is_enabled = state

    def set_logging(
        self,
        *,
        scalars: bool | None = None,
        histograms: bool | None = None,
    ) -> None:
        if scalars is not None:
            self.log_scalars = scalars
        if histograms is not None:
            self.log_histograms = histograms

    def add_scalar(self, tag: str, scalar_value: Any, global_step: int) -> None:
        if self.is_enabled and self.log_scalars and self.writer:
            self.writer.add_scalar(tag, scalar_value, global_step)

    def add_scalars(
        self, main_tag: str, tag_scalar_dict: dict, global_step: int
    ) -> None:
        if self.is_enabled and self.log_scalars and self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, global_step)

    def add_histogram(self, tag: str, values: Any, global_step: int) -> None:
        if self.is_enabled and self.log_histograms and self.writer:
            self.writer.add_histogram(tag, values, global_step)

        if self.is_enabled and self.log_histograms and self.writer:
            self.writer.add_histogram(tag, values, global_step)

    def log_weight_hist(self, tag: str, model: nn.Module, step: int) -> None:
        for name, param in model.named_parameters():
            self.add_histogram(f"{tag}/{name}", param, step)

    def log_grad_hist(self, tag: str, model: nn.Module, step: int) -> None:
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            self.add_histogram(f"{tag}/{name}", param.grad, step)

    @staticmethod
    def _convert_rewards_to_list(
        reward_info: dict[str, np.ndarray], num_envs: int
    ) -> list[dict[str, Any]]:
        list_rewards = [{} for _ in range(num_envs)]
        for key, value in reward_info.items():
            if key.startswith("_"):
                continue
            for env_num, reward_value in enumerate(value):
                list_rewards[env_num][key] = float(reward_value)
        return list_rewards

    def log_rewards(
        self,
        reward_info: dict[str, np.ndarray],
        valid_idx: np.ndarray,
        step: int,
        prefix: str = "transit/reward",
    ) -> None:
        list_rewards = self._convert_rewards_to_list(reward_info, len(valid_idx))
        for env_idx, reward_dict in enumerate(list_rewards):
            if valid_idx[env_idx] is False:
                continue

            tag = f"{prefix}/env_{env_idx}"
            self.add_scalars(tag, reward_dict, step)
            break

    def log_preds(
        self,
        reward_key: list[str],
        pred: torch.Tensor,
        step: int,
        prefix: str = "train/q",
    ) -> None:
        if len(reward_key) != pred.shape[1]:
            raise ValueError(
                f"Reward key length ({len(reward_key)}) does not match prediction shape ({pred.shape[1]})."
            )

        for idx, key in enumerate(reward_key):
            self.add_histogram(f"{prefix}/{key}", pred[:, idx], step)

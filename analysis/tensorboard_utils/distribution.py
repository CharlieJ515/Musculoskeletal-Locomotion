from typing import Any

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from .writer import get_writer


def log_weight_hist(tag: str, model: nn.Module, step: int):
    writer = get_writer()
    for name, param in model.named_parameters():
        writer.add_histogram(f"{tag}/{name}", param, step)


def log_grad_hist(tag: str, model: nn.Module, step: int):
    writer = get_writer()
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        writer.add_histogram(f"{tag}/{name}", param.grad, step)


def convert_rewards_to_list(
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
    reward_info: dict[str, np.ndarray],
    valid_idx: np.ndarray,
    step: int,
    prefix: str = "transit/reward",
):
    writer = get_writer()
    list_rewards = convert_rewards_to_list(reward_info, len(valid_idx))
    for env_idx, reward_dict in enumerate(list_rewards):
        if valid_idx[env_idx] is False:
            continue

        tag = f"{prefix}/env_{env_idx}"
        writer.add_scalars(tag, reward_dict, step)


def log_preds(
    reward_key: list[str], pred: torch.Tensor, step: float, prefix: str = "train/q"
):
    writer = get_writer()
    if len(reward_key) != pred.shape[1]:
        raise ValueError

    for idx, key in enumerate(reward_key):
        writer.add_histogram(f"{prefix}/{key}", pred[:, idx], step)

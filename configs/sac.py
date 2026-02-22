from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class SACConfig:
    state_dim: tuple[int, ...]
    action_dim: tuple[int, ...]
    actor_net: type[nn.Module]
    critic_net: type[nn.Module]
    target_entropy: float
    reward_dim: tuple[int, ...] = (1,)
    reward_weight: torch.Tensor = torch.ones(1)
    gamma: float = 0.99
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    lr: float = 3e-5
    tau: float = 1e-2
    weight_decay: float = 0.0
    policy_update_freq: int = 1
    device: Optional[torch.device] = None
    use_jit: bool = True
    train: bool = True
    name: str = "SAC"
    load_ckpt: bool = False
    ckpt_file: Optional[Path] = None

    @classmethod
    def from_dict(
        cls, data: dict, actor_net: type[nn.Module], critic_net: type[nn.Module]
    ):
        d = data.copy()

        d["state_dim"] = tuple(d["state_dim"])
        d["action_dim"] = tuple(d["action_dim"])
        d["reward_dim"] = tuple(d.get("reward_dim", [1]))

        d["actor_net"] = actor_net
        d["critic_net"] = critic_net

        if "reward_weight" in d:
            d["reward_weight"] = torch.tensor(d["reward_weight"], dtype=torch.float32)
        if d.get("device"):
            d["device"] = torch.device(d["device"])
        if d.get("ckpt_file"):
            d["ckpt_file"] = Path(d["ckpt_file"])

        return cls(**d)

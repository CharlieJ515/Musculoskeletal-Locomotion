from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class TD3Config:
    state_dim: tuple[int, ...]
    action_dim: tuple[int, ...]
    actor_net: type[nn.Module]
    critic_net: type[nn.Module]
    reward_dim: tuple[int, ...] = (1,)
    reward_weight: torch.Tensor = torch.ones(1)
    gamma: float = 0.99
    lr: float = 3e-4
    tau: float = 0.005
    weight_decay: float = 0.0
    policy_update_freq: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    max_action: float = 1.0
    device: Optional[torch.device] = None
    use_jit: bool = True
    train: bool = True
    name: str = "TD3"
    load_chkpt: bool = False
    chkpt_file: Optional[Path] = None

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
        if d.get("chkpt_file"):
            d["chkpt_file"] = Path(d["chkpt_file"])

        return cls(**d)

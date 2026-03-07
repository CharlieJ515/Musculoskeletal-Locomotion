from dataclasses import dataclass

import torch


@dataclass(kw_only=True)
class ReplayBufferConfig:
    capacity: int
    obs_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    reward_shape: tuple[int, ...] = (1,)
    device: torch.device = torch.device("cpu")
    obs_dtype: torch.dtype = torch.float32
    action_dtype: torch.dtype = torch.float32
    reward_dtype: torch.dtype = torch.float32
    name: str = "ReplayBuffer"

    @classmethod
    def from_dict(cls, data: dict):
        d = data.copy()

        d["obs_shape"] = tuple(d["obs_shape"])
        d["action_shape"] = tuple(d["action_shape"])
        d["reward_shape"] = tuple(d.get("reward_shape", [1]))
        if "device" in d:
            d["device"] = torch.device(d["device"])

        for dtype_key in ["obs_dtype", "action_dtype", "reward_dtype"]:
            if dtype_key not in d:
                continue
            d[dtype_key] = getattr(torch, d[dtype_key])

        return cls(**d)


@dataclass(kw_only=True)
class PERConfig(ReplayBufferConfig):
    beta_frames: int
    alpha: float = 0.6
    beta_start: float = 0.4
    epsilon: float = 1e-5
    name: str = "PrioritizedReplayBuffer/"

    @classmethod
    def from_dict(cls, data: dict):
        d = data.copy()

        d["obs_shape"] = tuple(d["obs_shape"])
        d["action_shape"] = tuple(d["action_shape"])
        d["reward_shape"] = tuple(d.get("reward_shape", [1]))
        if "device" in d:
            d["device"] = torch.device(d["device"])

        for dtype_key in ["obs_dtype", "action_dtype", "reward_dtype"]:
            if dtype_key not in d:
                continue
            d[dtype_key] = getattr(torch, d[dtype_key])

        return cls(**d)

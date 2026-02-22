from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import mlflow
import numpy as np
import torch
import torch.nn as nn

from environment.models import MODEL_REGISTRY
from environment.osim.pose import POSE_REGISTRY, Pose

from .noise import NoiseConfig


@dataclass
class TrainConfig:
    total_steps: int
    start_random: int
    batch_size: int
    eval_interval: int
    eval_episodes: int
    log_interval: int
    seed: int

    model: Path
    pose: Pose
    visualize: bool
    num_env: int
    mp_context: Literal["spawn", "fork", "forkserver"]

    noise: NoiseConfig
    wrappers: list[dict[str, Any]] = field(default_factory=list)
    rewards: list[dict[str, Any]] = field(default_factory=list)
    reward_key: list[str] = field(default_factory=list)

    def log_params(self):
        mlflow.log_params(
            {
                "total_steps": self.total_steps,
                "start_random": self.start_random,
                "batch_size": self.batch_size,
                "eval_interval": self.eval_interval,
                "eval_episodes": self.eval_episodes,
                "log_interval": self.log_interval,
                "seed": self.seed,
                "init_pose": self.pose.name,
                "opensim_model": str(self.model),
                "visualize": self.visualize,
                "num_env": self.num_env,
                "reward_key": self.reward_key,
                "mp_context": self.mp_context,
            }
        )

    def __post_init__(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if not self.reward_key and self.rewards:
            self.reward_key = [r.get("key", r["name"]) for r in self.rewards]

    @classmethod
    def from_dict(cls, data: dict):
        d = data.copy()

        pose_name = d.get("pose")
        if pose_name not in POSE_REGISTRY:
            raise ValueError(f"Unknown pose '{pose_name}'")
        d["pose"] = POSE_REGISTRY[pose_name]()

        model_name = d.get("model")
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model '{model_name}'")
        d["model"] = MODEL_REGISTRY[model_name]

        d["noise"] = NoiseConfig(**d["noise"])

        return cls(**d)

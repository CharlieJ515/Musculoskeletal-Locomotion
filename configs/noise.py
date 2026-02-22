from dataclasses import dataclass
from typing import Literal


@dataclass
class NoiseConfig:
    name: Literal["GaussianNoise", "OUNoise"]
    sigma: float
    sigma_min: float = 0.05
    decay_steps: int = 50_000
    theta: float = 0.15
    dt: float = 1e-2
    clip: float | None = None

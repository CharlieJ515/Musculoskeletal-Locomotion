from abc import ABC, abstractmethod

import numpy as np

from configs import NoiseConfig


class BaseNoise(ABC):
    def __init__(
        self,
        batch_size: int,
        action_dim: tuple[int, ...],
        sigma_start: float = 0.4,
        sigma_min: float = 0.05,
        decay_steps: int = 50000,
    ):
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.sigma_start = sigma_start
        self.sigma_min = sigma_min
        self.decay_steps = decay_steps
        self.current_step = 0

    @property
    def current_sigma(self) -> float:
        if self.decay_steps == 0:
            return self.sigma_start

        # Calculate linear decay rate bounded between 0 and 1
        decay_rate = min(1.0, self.current_step / self.decay_steps)
        return self.sigma_start - decay_rate * (self.sigma_start - self.sigma_min)

    @abstractmethod
    def sample(self) -> np.ndarray:
        pass

    @abstractmethod
    def reset(self, mask: np.ndarray | None = None):
        pass


class GaussianNoise(BaseNoise):
    def __init__(
        self,
        batch_size: int,
        action_dim: tuple[int, ...],
        sigma_start: float = 0.4,
        sigma_min: float = 0.05,
        decay_steps: int = 50000,
        clip: float | None = None,
    ):
        super().__init__(batch_size, action_dim, sigma_start, sigma_min, decay_steps)
        self.clip = clip

    def sample(self) -> np.ndarray:
        shape = (self.batch_size, *self.action_dim)
        noise = np.random.normal(0, self.current_sigma, size=shape)

        if self.clip is not None:
            noise = np.clip(noise, -self.clip, self.clip)

        self.current_step += 1
        return noise

    def reset(self, *args, **kwargs):
        pass


class OUNoise(BaseNoise):
    def __init__(
        self,
        batch_size: int,
        action_dim: tuple[int, ...],
        theta: float = 0.15,
        sigma_start: float = 0.4,
        sigma_min: float = 0.05,
        decay_steps: int = 50000,
        dt: float = 1e-2,
    ):
        super().__init__(batch_size, action_dim, sigma_start, sigma_min, decay_steps)
        self.theta = theta
        self.dt = dt
        self.state = np.zeros((self.batch_size, *self.action_dim), dtype=np.float32)

    def reset(self, mask: np.ndarray | None = None):
        if mask is None:
            self.state[...] = 0.0
            return

        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != self.batch_size:
            raise ValueError("mask must have shape (batch_size,)")

        self.state[mask] = 0.0

    def sample(self) -> np.ndarray:
        x = self.state
        dW = np.random.randn(*x.shape).astype(np.float32)

        dx = -self.theta * x * self.dt + self.current_sigma * np.sqrt(self.dt) * dW
        self.state = x + dx

        self.current_step += 1
        return self.state


NOISE_REGISTRY: dict[str, type[BaseNoise]] = {
    "GaussianNoise": GaussianNoise,
    "OUNoise": OUNoise,
}


def build_noise(
    cfg: NoiseConfig, batch_size: int, action_dim: tuple[int, ...]
) -> BaseNoise:
    if cfg.name not in NOISE_REGISTRY:
        raise ValueError(f"Unknown noise generator: {cfg.name}")

    NoiseClass = NOISE_REGISTRY[cfg.name]

    kwargs = {
        "batch_size": batch_size,
        "action_dim": action_dim,
        "sigma_start": cfg.sigma,
        "sigma_min": cfg.sigma_min,
        "decay_steps": cfg.decay_steps,
    }

    if cfg.name == "OUNoise":
        kwargs["theta"] = cfg.theta
        kwargs["dt"] = cfg.dt
    elif cfg.name == "GaussianNoise":
        kwargs["clip"] = cfg.clip

    return NoiseClass(**kwargs)

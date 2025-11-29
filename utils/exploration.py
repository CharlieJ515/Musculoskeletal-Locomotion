import numpy as np


class GaussianNoise:
    def __init__(
        self,
        batch_size: int,
        action_dim: tuple[int, ...],
        start_std: float,
        end_std: float,
        decay_steps: int,
        clip: float | None = None,
    ):
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.start_std = start_std
        self.end_std = end_std
        self.decay_steps = decay_steps
        self.clip = clip

        self._t = 0

    def sample(self) -> np.ndarray:
        std = self.get_current_std()
        shape = (self.batch_size, *self.action_dim)
        noise = np.random.normal(0, std, size=shape)

        if self.clip is not None:
            noise = np.clip(noise, -self.clip, self.clip)

        self._t += 1

        return noise

    def get_current_std(self) -> float:
        progress = min(1.0, self._t / self.decay_steps)
        return self.start_std - (self.start_std - self.end_std) * progress


class OUNoise:
    def __init__(
        self,
        batch_size: int,
        action_dim: tuple[int, ...],
        theta: float = 0.15,
        sigma: float = 0.1,
        dt: float = 1e-2,
    ):
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.theta = theta
        self.sigma = sigma
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

        dx = -self.theta * x * self.dt + self.sigma * np.sqrt(self.dt) * dW
        self.state = x + dx

        return self.state

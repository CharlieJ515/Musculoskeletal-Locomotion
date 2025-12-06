import numpy as np


class GaussianNoise:
    def __init__(
        self,
        batch_size: int,
        action_dim: tuple[int, ...],
        sigma: float,
        clip: float | None = None,
    ):
        self.batch_size = batch_size
        self.action_dim = action_dim
        self.sigma = sigma
        self.clip = clip

    def sample(self) -> np.ndarray:
        shape = (self.batch_size, *self.action_dim)
        noise = np.random.normal(0, self.sigma, size=shape)

        if self.clip is not None:
            noise = np.clip(noise, -self.clip, self.clip)

        return noise

    def reset(self, *args, **kwargs):
        pass


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

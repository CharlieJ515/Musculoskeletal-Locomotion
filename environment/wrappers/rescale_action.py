from typing import TypeVar, Literal

import gymnasium as gym
import numpy as np

ObsType = TypeVar("ObsType")


class RescaleActionWrapper(gym.ActionWrapper[ObsType, np.ndarray, np.ndarray]):
    def __init__(
        self, env: gym.Env[ObsType, np.ndarray], mode: Literal["abs", "square"]
    ):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box)
        assert np.all(env.action_space.low == 0.0), "Env action_space.low must be 0"
        assert np.all(env.action_space.high == 1.0), "Env action_space.high must be 1"

        self.mode = mode

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=env.action_space.shape,
            dtype=np.float32,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        if self.mode == "abs":
            return np.abs(action)

        return np.square(action)

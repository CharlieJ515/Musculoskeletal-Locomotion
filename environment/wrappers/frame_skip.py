from typing import Optional, TypeVar, Any, SupportsFloat
from pathlib import Path
import warnings

import gymnasium as gym
import opensim

from environment.osim import OsimEnv
from .composite_reward import CompositeRewardWrapper
from .utils import has_wrapper

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class FrameSkipWrapper(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        skip: int = 4,
    ):
        super().__init__(env)
        if skip < 1:
            raise ValueError("Expected skip >= 1")

        self.skip = skip
        self.has_composite_reward = has_wrapper(env, CompositeRewardWrapper)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        rewards: list[dict[str, float]] = []
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)

            total_reward += float(reward)
            if self.has_composite_reward:
                rewards.append(info["rewards"])

            if terminated or truncated:
                break

        if not self.has_composite_reward:
            return obs, total_reward, terminated, truncated, info  # type: ignore

        composite_reward = {}
        for key in rewards[0].keys():
            reward = 0.0
            for r in rewards:
                reward += r[key]

            composite_reward[key] = reward

        info["rewards"] = composite_reward  # type: ignore
        return obs, total_reward, terminated, truncated, info  # type: ignore

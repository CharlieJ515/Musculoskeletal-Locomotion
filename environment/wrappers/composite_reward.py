from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
import opensim

from environment.osim import Action, Observation
from environment.rewards import CompositeReward


def reward_info_to_ndarray(
    reward_key: list[str], reward_info: dict[str, np.ndarray]
) -> np.ndarray:
    reward = np.array(
        [reward_info[key] for key in reward_key],
        dtype=np.float32,
    ).T
    return reward


class CompositeRewardWrapper(gym.Wrapper[Observation, Action, Observation, Action]):
    def __init__(
        self,
        env: gym.Env[Observation, Action],
        reward_fn: CompositeReward,
        *,
        base_reward_weight: float = 1.0,
    ):
        super().__init__(env)
        self._reward_fn = reward_fn
        self._base_reward_weight = base_reward_weight

    def _get_model(self) -> opensim.Model:
        base_env: OsimEnv = self.unwrapped  # type: ignore[reportAssignmentType]
        return base_env.model

    def _get_state(self) -> opensim.State:
        base_env: OsimEnv = self.unwrapped  # type: ignore[reportAssignmentType]
        return base_env.state

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        if obs.normalized:
            raise RuntimeError(
                "Received normalized observation, run normalization at last"
            )

        model = self._get_model()
        state = self._get_state()
        self._reward_fn.reset(model, state, obs)

        return obs, info

    def step(
        self, action: Action
    ) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, base_reward, terminated, truncated, info = self.env.step(action)
        base_reward = float(base_reward)

        model = self._get_model()
        state = self._get_state()

        composite_reward, rewards = self._reward_fn.compute(
            model,
            state,
            obs,
            action,
            terminated,
            truncated,
        )
        total_reward = self._base_reward_weight * base_reward + composite_reward
        rewards["base"] = base_reward
        rewards["total"] = total_reward
        info["rewards"] = rewards

        return obs, total_reward, terminated, truncated, info

from typing import TypeVar, Any, SupportsFloat, Tuple
import dataclasses
import math

import gymnasium as gym

from environment.osim import Observation
from utils.vec3 import Vec3

ActType = TypeVar("ActType")


class TargetSpeedWrapper(gym.Wrapper[Observation, ActType, Observation, ActType]):
    def __init__(
        self,
        env: gym.Env[Observation, ActType],
        *,
        speed_range: Tuple[float, float] = (1.2, 1.8),
        speed_tolerance: float = 0.10,  # target speed tolerance
        hold_steps: int = 150,
    ):
        super().__init__(env)
        self.speed_range = speed_range
        self.speed_tolerance = speed_tolerance
        self.hold_steps = hold_steps

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

        self.hold_counter: int = 0
        self.target_speed: float = self.np_random.uniform(*self.speed_range)
        self.target_speed = round(self.target_speed, 1)

        # target velocity is relative to the pelvis
        target_vel = Vec3(self.target_speed, 0, 0)
        obs = self._set_target_to_obs(obs, target_vel)
        info["target_speed"] = self.target_speed
        return obs, info

    @staticmethod
    def _set_target_to_obs(obs: Observation, target_vel: Vec3) -> Observation:
        return dataclasses.replace(obs, target_velocity=target_vel)

    def step(
        self, action: ActType
    ) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        pelvis = obs.body["pelvis"]
        pelvis_vel = pelvis.vel
        pelvis_speed = math.hypot(pelvis_vel.x, pelvis_vel.z)
        err = pelvis_speed - self.target_speed
        reward = -(err**2)

        if abs(err) <= self.speed_tolerance:
            self.hold_counter = self.hold_counter + 1
            terminated = self.hold_counter >= self.hold_steps or terminated
            info["success"] = terminated
        else:
            self.hold_counter = 0

        target_vel = Vec3(self.target_speed, 0, 0)
        obs = self._set_target_to_obs(obs, target_vel)
        info["target_speed"] = self.target_speed
        return obs, reward, terminated, truncated, info

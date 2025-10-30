from typing import TypeVar, Any, SupportsFloat, Tuple
import dataclasses
import warnings
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

    @staticmethod
    def _set_target_to_obs(obs: Observation, target_vel: Vec3) -> Observation:
        return dataclasses.replace(obs, target_velocity=target_vel)

    def _sample_target(self):
        self.target_speed: float = round(self.np_random.uniform(*self.speed_range), 1)
        self.target_vel = Vec3(self.target_speed, 0, 0)

    def _get_distance(self, vel: Vec3, yaw: float) -> float:
        vel_rot = vel.rotate_y(-yaw)
        difference = self.target_vel - vel_rot
        distance = difference.magnitude()

        return distance

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
        self._sample_target()

        pelvis_vel = dataclasses.replace(obs.pelvis.vel, y=0)
        yaw = obs.pelvis.ang.y
        dist = self._get_distance(pelvis_vel, yaw)

        in_range = dist <= self.speed_tolerance
        success = self.hold_counter >= self.hold_steps

        obs = self._set_target_to_obs(obs, self.target_vel)
        info["target"] = {
            "in_range": in_range,
            "hold_counter": self.hold_counter,
            "success": success,
            "target_speed": self.target_speed,
            "current_speed": pelvis_vel.magnitude(),
        }

        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward != 0:
            warnings.warn(
                "Reward from base environment is being overwritten with reward from TargetSpeedWrapper"
            )

        pelvis_vel = dataclasses.replace(obs.pelvis.vel, y=0)
        yaw = obs.pelvis.ang.y
        dist = self._get_distance(pelvis_vel, yaw)

        in_range = dist <= self.speed_tolerance
        self.hold_counter = self.hold_counter + 1 if in_range else 0
        success = self.hold_counter >= self.hold_steps
        terminated = terminated or success

        obs = self._set_target_to_obs(obs, self.target_vel)
        info["target"] = {
            "in_range": in_range,
            "hold_counter": self.hold_counter,
            "success": success,
            "target_speed": self.target_speed,
            "current_speed": pelvis_vel.magnitude(),
        }
        return obs, 0.0, terminated, truncated, info

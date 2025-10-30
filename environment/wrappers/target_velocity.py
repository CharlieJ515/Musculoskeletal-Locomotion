from typing import TypeVar, Any, Tuple, SupportsFloat
import dataclasses
import math
import warnings

import gymnasium as gym

from environment.osim import Observation
from utils.vec3 import Vec3

ActType = TypeVar("ActType")


class TargetVelocityWrapper(gym.Wrapper[Observation, ActType, Observation, ActType]):
    def __init__(
        self,
        env: gym.Env[Observation, ActType],
        *,
        box_x: Tuple[float, float] = (-10.0, 10.0),
        box_z: Tuple[float, float] = (-10.0, 10.0),
        max_speed: float = 1.5,  # 5.4 km/h, a decent walking speed
        decel_radius: float = 0.5,  # distance from target to start deceleration
        r_target: float = 0.3,
        hold_steps: int = 50,
    ):
        super().__init__(env)
        self.box_x = box_x
        self.box_z = box_z
        self.max_speed = round(max_speed, 1)
        self.decel_radius = round(decel_radius, 1)
        self.r_target = r_target
        self.hold_steps = hold_steps

    @staticmethod
    def _set_target_to_obs(obs: Observation, target_vel_local: Vec3) -> Observation:
        return dataclasses.replace(obs, target_velocity=target_vel_local)

    def _speed_from_distance(self, d: float) -> float:
        if d > self.decel_radius:
            return self.max_speed

        # within decel radius — linearly drop to zero
        return round((d / self.decel_radius) * self.max_speed, 1)

    def _sample_target(self):
        target_x = round(self.np_random.uniform(*self.box_x), 1)
        target_z = round(self.np_random.uniform(*self.box_z), 1)
        self.target = Vec3(target_x, 0, target_z)

    def _get_target(self, pos: Vec3, yaw: float) -> tuple[Vec3, float, float]:
        if pos.y != 0:
            raise ValueError
        difference = self.target - pos
        distance = difference.magnitude()

        if distance < 0.1:
            return Vec3(0.0, 0.0, 0.0), distance, 0.0

        direction = difference.norm().rotate_y(-yaw)
        target_speed = self._speed_from_distance(distance)
        return direction * target_speed, distance, target_speed

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> Tuple[Observation, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        if obs.normalized:
            raise RuntimeError(
                "Received normalized observation, run normalization at last"
            )

        self.hold_counter = 0
        self._sample_target()

        pos = dataclasses.replace(obs.pelvis.pos, y=0)
        yaw = obs.pelvis.ang.y

        target_vel, dist, target_speed = self._get_target(pos, yaw)
        in_range = dist <= self.r_target
        success = self.hold_counter >= self.hold_steps

        obs = self._set_target_to_obs(obs, target_vel)
        info["target"] = {
            "in_range": in_range,
            "hold_counter": self.hold_counter,
            "success": success,
            "distance_to_target": dist,
            "target_speed": target_speed,
            "target_position": self.target,
        }
        return obs, info

    def step(
        self, action: ActType
    ) -> Tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward != 0:
            warnings.warn(
                "Reward from base environment is being overwritten with reward from TargetVelocityWrapper"
            )

        pos = dataclasses.replace(obs.pelvis.pos, y=0)
        yaw = obs.pelvis.ang.y
        target_vel, dist, target_speed = self._get_target(pos, yaw)

        in_range = dist <= self.r_target
        self.hold_counter = self.hold_counter + 1 if in_range else 0
        success = self.hold_counter >= self.hold_steps
        terminated = terminated or success

        reward = -dist
        obs = self._set_target_to_obs(obs, target_vel)
        info["target"] = {
            "in_range": in_range,
            "hold_counter": self.hold_counter,
            "success": success,
            "distance_to_target": dist,
            "target_speed": target_speed,
            "target_position": self.target,
        }
        return obs, reward, terminated, truncated, info

from typing import TypeVar, Any, Tuple, SupportsFloat
import dataclasses
import math
import warnings

import gymnasium as gym

from environment.osim import Observation
from utils.vec3 import Vec3

ActType = TypeVar("ActType")


def rotate_frame(x: float, z: float, yaw: float) -> Tuple[float, float]:
    x_rot = math.cos(yaw) * x - math.sin(yaw) * z
    z_rot = math.sin(yaw) * x + math.cos(yaw) * z
    return x_rot, z_rot


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
        self.target_x = round(self.np_random.uniform(*self.box_x), 1)
        self.target_z = round(self.np_random.uniform(*self.box_z), 1)

    def _get_target(self, x: float, z: float, yaw: float) -> tuple[Vec3, float, float]:
        dx = self.target_x - x
        dz = self.target_z - z
        dist = math.hypot(dx, dz)

        if dist < 0.1:
            return Vec3(0.0, 0.0, 0.0), dist, 0.0

        dir_wx, dir_wz = dx / dist, dz / dist
        vf, vl = rotate_frame(dir_wx, dir_wz, yaw)
        speed = self._speed_from_distance(dist)
        return Vec3(vf * speed, 0.0, vl * speed), dist, speed

    def _success_metrics(
        self,
        dist: float,
        update_counter: bool = True,
    ) -> Tuple[bool, bool]:
        in_range = dist <= self.r_target
        self.hold_counter = self.hold_counter + 1 if in_range and update_counter else 0
        reached = self.hold_counter >= self.hold_steps
        return in_range, reached

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

        pelvis = obs.body["pelvis"]
        x, z = pelvis.pos.x, pelvis.pos.z
        yaw = pelvis.ang.y

        target_vel, dist, target_speed = self._get_target(x, z, yaw)
        in_range, success = self._success_metrics(dist, update_counter=False)

        obs = self._set_target_to_obs(obs, target_vel)
        info.update(
            {
                "distance_to_target": dist,
                "in_range": in_range,
                "hold_counter": self.hold_counter,
                "success": success,
                "target_speed": target_speed,
                "target_position_xz": (self.target_x, self.target_z),
                "hold_steps": self.hold_steps,
                "r_target": self.r_target,
            }
        )
        return obs, info

    def step(
        self, action: ActType
    ) -> Tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward != 0:
            warnings.warn(
                "Reward from base environment is being overwritten with reward from TargetVelocityWrapper"
            )

        pelvis = obs.body["pelvis"]
        x, z = pelvis.pos.x, pelvis.pos.z
        yaw = pelvis.ang.y

        target_vel, dist, target_speed = self._get_target(x, z, yaw)
        in_range, success = self._success_metrics(dist)

        reward = -dist
        terminated = terminated or success
        obs = self._set_target_to_obs(obs, target_vel)
        info.update(
            {
                "distance_to_target": dist,
                "in_range": in_range,
                "hold_counter": self.hold_counter,
                "success": success,
                "target_speed": target_speed,
                "target_position_xz": (self.target_x, self.target_z),
            }
        )
        return obs, reward, terminated, truncated, info

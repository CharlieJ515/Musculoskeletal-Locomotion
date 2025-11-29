from typing import TypeVar, Any
from pathlib import Path
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import opensim

from environment.osim import OsimEnv, Observation, OsimModel


ActType = TypeVar("ActType")


@dataclass
class LimitForceConfig:
    coordinate_name: str
    upper_limit: float
    upper_stiffness: float
    lower_limit: float
    lower_stiffness: float
    damping: float
    transition: float
    dissipate_energy: bool


class BabyWalkerWrapper(gym.Wrapper[Observation, ActType, Observation, ActType]):
    def __init__(
        self,
        env: gym.Env,
        limits: list[LimitForceConfig],
        *,
        reward_scale: float = 1.0,
        decay_steps: float = 30_000,
    ):
        super().__init__(env)
        self.base_name = "baby_walker"
        self.reward_scale = reward_scale
        self.decay_steps = decay_steps
        self.total_step = 0

        self.limits: dict[str, LimitForceConfig] = {}
        self.limit_index: dict[str, int] = {}
        self._inject_limits(limits)

    def _get_model(self) -> opensim.Model:
        base_env: OsimEnv = self.unwrapped  # type: ignore[reportAssignmentType]
        return base_env.model

    def _init_system(self):
        base_env: OsimEnv = self.unwrapped  # type: ignore[reportAssignmentType]
        osim_model: OsimModel = base_env.osim_model
        osim_model.init_system()

    def _inject_limits(self, limits: list[LimitForceConfig]):
        model = self._get_model()
        force_set: opensim.ForceSet = model.getForceSet()

        for c in limits:
            limit_name = f"{self.base_name}_{c.coordinate_name}"
            if force_set.contains(limit_name):
                continue

            limit = opensim.CoordinateLimitForce()
            limit.setName(limit_name)
            limit.set_coordinate(c.coordinate_name)
            limit.set_upper_limit(c.upper_limit)
            limit.set_upper_stiffness(c.upper_stiffness)
            limit.set_lower_limit(c.lower_limit)
            limit.set_lower_stiffness(c.lower_stiffness)
            limit.set_damping(c.damping)
            limit.set_transition(c.transition)
            limit.set_compute_dissipation_energy(c.dissipate_energy)

            model.addForce(limit)
            self.limits[limit_name] = c
            self.limit_index[limit_name] = force_set.getSize() - 1

        # restart to apply modifications
        self._init_system()

    def step(
        self, action: ActType
    ) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_step += 1

        penalty = 0.0
        bodyweight = obs.norm_spec.mass * obs.norm_spec.g  # type: ignore
        forces: dict[str, float] = {}
        for name, c in self.limits.items():
            force_state = obs.force[name]
            f = force_state[name]

            forces[name] = f
            penalty += abs(f) / bodyweight

        info["baby_walker"] = {
            "penalty": penalty,
            "forces": forces,
        }

        total_reward = float(reward) - penalty * self.reward_scale
        return obs, total_reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        factor = 1.0
        if self.decay_steps != 0:
            progress = min(self.total_step / self.decay_steps, 1.0)
            factor = max(factor - progress, 1e-5)
        self._update_stiffness(factor)

        obs, info = self.env.reset(seed=seed, options=options)
        return obs, info

    def _update_stiffness(self, factor: float):
        model = self._get_model()
        force_set = model.getForceSet()

        for name, cfg in self.limits.items():
            index = self.limit_index[name]
            force: opensim.Force = force_set.get(index)
            limit_force: opensim.CoordinateLimitForce = (
                opensim.CoordinateLimitForce.safeDownCast(force)
            )

            new_upper = cfg.upper_stiffness * factor
            new_lower = cfg.lower_stiffness * factor

            limit_force.set_upper_stiffness(new_upper)
            limit_force.set_lower_stiffness(new_lower)

        self._init_system()


def plot_limit_force(
    env: BabyWalkerWrapper, coordinate_name: str, interval: float = 0.1
):
    limit_name = f"{env.base_name}_{coordinate_name}"
    if limit_name not in env.limits:
        raise ValueError(f"{limit_name} not in {env.limits.keys()}")
    limit = env.limits[limit_name]

    base_env: OsimEnv = env.unwrapped  # type: ignore
    model = base_env.model
    state = base_env.state

    coord_set: opensim.CoordinateSet = model.getCoordinateSet()
    coord: opensim.Coordinate = coord_set.get(limit.coordinate_name)
    coord_min: float = coord.get_range(0)
    coord_max: float = coord.get_range(1)

    force_set: opensim.ForceSet = model.getForceSet()
    force: opensim.Force = force_set.get(limit_name)
    limit_force: opensim.CoordinateLimitForce = (
        opensim.CoordinateLimitForce.safeDownCast(force)
    )

    coord_values = np.arange(coord_min, coord_max, interval)
    force_values = np.zeros_like(coord_values)

    for idx, c in enumerate(coord_values):
        coord.setValue(state, c)
        model.realizeAcceleration(state)

        f = limit_force.calcLimitForce(state)
        force_values[idx] = f

    return coord_values, force_values

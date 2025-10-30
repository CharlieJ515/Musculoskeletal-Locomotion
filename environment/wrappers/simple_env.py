from typing import List, Any, SupportsFloat

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import opensim

from environment.osim import Observation, Action, OsimEnv


def flatten_observation(obs: Observation) -> np.ndarray:
    parts: List[float] = []

    for j in obs.joint.values():
        parts.extend(j.ang)
        parts.extend(j.ang_vel)
        parts.extend(j.ang_acc)

    for b in obs.body.values():
        parts.extend(b.pos.to_tuple())
        parts.extend(b.vel.to_tuple())
        parts.extend(b.acc.to_tuple())
        parts.extend(b.ang.to_tuple())
        parts.extend(b.ang_vel.to_tuple())
        parts.extend(b.ang_acc.to_tuple())

    parts.extend(obs.pelvis.pos.to_tuple())
    parts.extend(obs.pelvis.vel.to_tuple())
    parts.extend(obs.pelvis.acc.to_tuple())
    parts.extend(obs.pelvis.ang.to_tuple())
    parts.extend(obs.pelvis.ang_vel.to_tuple())
    parts.extend(obs.pelvis.ang_acc.to_tuple())

    for m in obs.muscle.values():
        parts.append(m.activation)
        parts.append(m.fiber_length)
        parts.append(m.fiber_velocity)
        parts.append(m.fiber_force)

    for f in obs.foot.values():
        for comp in (f.ground, f.calcn, f.toes):
            parts.extend(comp.force.to_tuple())
            parts.extend(comp.torque.to_tuple())

    parts.extend(obs.target_velocity.to_tuple())

    return np.asarray(parts, dtype=np.float64)


def build_action_space() -> spaces.Box:
    length = len(Action.muscle_order)
    assert length > 0, "No muscles found: cannot construct action space."

    return spaces.Box(0.0, 1.0, (length,), dtype=np.float32)


def build_observation_space(model: opensim.Model) -> spaces.Box:
    length = 0

    joint_set = model.getJointSet()
    for i in range(joint_set.getSize()):
        j = joint_set.get(i)
        name = j.getName()
        dof = j.numCoordinates()
        length += dof * 3

    body_set = model.getBodySet()
    length += body_set.getSize() * 18
    length += 18  # pelvis

    muscles = model.getMuscles()
    length += muscles.getSize() * 4

    force_set = model.getForceSet()
    for i in range(force_set.getSize()):
        f = force_set.get(i)
        name = f.getName()
        if not name.startswith("foot_"):
            continue

        length += 18

    length += 3  # target_velocity_space

    obs_space = spaces.Box(-np.inf, np.inf, (length,), dtype=np.float32)
    return obs_space


class SimpleEnvWrapper(gym.Wrapper[np.ndarray, np.ndarray, Observation, Action]):
    def __init__(self, env: gym.Env[Observation, Action]):
        super().__init__(env)

        base_env: OsimEnv = env.unwrapped  # type: ignore[reportAssignmentType]
        model = base_env.model
        if len(Action.muscle_order) == 0:
            muscle_order = Action.build_muscle_order(model)
            Action.muscle_order = muscle_order

        self.action_space = build_action_space()
        self.observation_space = build_observation_space(model)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        obs_flatten = flatten_observation(obs)

        return obs_flatten, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        action_ = Action.from_numpy(action)
        obs, reward, terminated, truncated, info = self.env.step(action_)
        obs_flatten = flatten_observation(obs)
        return obs_flatten, reward, terminated, truncated, info

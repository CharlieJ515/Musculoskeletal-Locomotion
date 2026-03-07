import dataclasses

import opensim

from environment.osim.action import Action
from environment.osim.observation import Observation

from .base import RewardComponent


class AliveReward(RewardComponent):
    __slots__ = ["alive_reward", "failure_penalty"]

    def __init__(self, alive_reward: float = 1.0, failure_penalty: float = -200.0):
        self.alive_reward = alive_reward
        self.failure_penalty = failure_penalty

    def compute(
        self,
        model: opensim.Model,
        state: opensim.State,
        obs: Observation,
        action: Action,
        terminated: bool,
        truncated: bool,
    ) -> float:
        if terminated and not truncated:
            return self.failure_penalty

        return self.alive_reward


class VelocityReward(RewardComponent):
    __slots__ = ["scale"]

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def compute(
        self,
        model: opensim.Model,
        state: opensim.State,
        obs: Observation,
        action: Action,
        terminated: bool,
        truncated: bool,
    ) -> float:
        vel = dataclasses.replace(obs.pelvis.vel, y=0)
        yaw = obs.pelvis.ang.y
        vel_rot = vel.rotate_y(-yaw)
        target_vel = obs.target_velocity
        penalty = -(target_vel - vel_rot).magnitude()
        return penalty * self.scale


class FootstepReward(RewardComponent):
    def __init__(self, scale: float = 1.0, stepsize: float = 0.01):
        self.scale = scale
        self.stepsize = stepsize

        self._r_contact_prev = False
        self._l_contact_prev = False
        self._del_t = 0.0

    def reset(self, model: opensim.Model, state: opensim.State, obs: Observation):
        mass = obs.norm_spec.mass  # type: ignore
        g = obs.norm_spec.g  # type: ignore
        self.threshold = -0.05 * (mass * g)

        r_contact, l_contact = self._get_contacts(obs)
        self._r_contact_prev = r_contact
        self._l_contact_prev = l_contact
        self._del_t = 0.0

    def compute(
        self,
        model: opensim.Model,
        state: opensim.State,
        obs: Observation,
        action: Action,
        terminated: bool,
        truncated: bool,
    ) -> float:
        self._del_t += self.stepsize

        r_contact, l_contact = self._get_contacts(obs)
        new_step = (not self._r_contact_prev and r_contact) or (
            not self._l_contact_prev and l_contact
        )
        self._r_contact_prev, self._l_contact_prev = r_contact, l_contact

        if not new_step:
            return 0.0

        reward = self.scale * self._del_t
        self._del_t = 0.0
        return reward

    def _get_contacts(self, obs: Observation):
        r_force = obs.foot["foot_r"].ground.force.y
        l_force = obs.foot["foot_l"].ground.force.y

        r_contact = r_force < self.threshold
        l_contact = l_force < self.threshold
        return r_contact, l_contact

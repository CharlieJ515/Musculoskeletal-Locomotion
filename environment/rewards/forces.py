from typing import Dict

import numpy as np
import opensim

from environment.osim.action import Action
from environment.osim.observation import FootState, Observation

from .base import RewardComponent


class FootImpactPenalty(RewardComponent):
    __slots__ = ["stepsize", "scale", "_prev_foot"]

    def __init__(self, stepsize: float, scale: float = 1.0):
        self.stepsize = stepsize
        self.scale = scale

    def reset(self, model: opensim.Model, state: opensim.State, obs: Observation):
        self._prev_foot: Dict[str, FootState] = obs.foot

    def compute(
        self,
        model: opensim.Model,
        state: opensim.State,
        obs: Observation,
        action: Action,
        terminated: bool,
        truncated: bool,
    ) -> float:
        cost = 0.0
        current_foot = obs.foot
        for name in current_foot.keys():
            cur = current_foot[name].ground.force.y
            prev = self._prev_foot[name].ground.force.y
            dFdt = max(0, cur - prev) / self.stepsize
            cost += dFdt

        self._prev_foot = current_foot
        return -cost * self.scale


class BodySupportReward(RewardComponent):
    __slots__ = ["scale", "sensitivity"]

    def __init__(self, scale: float = 1.0, sensitivity: float = -10.0):
        self.scale = scale
        self.sensitivity = sensitivity

    def compute(
        self,
        model: opensim.Model,
        state: opensim.State,
        obs: Observation,
        action: Action,
        terminated: bool,
        truncated: bool,
    ) -> float:
        # ground vertical force is negative
        f_left = abs(obs.foot["foot_l"].ground.force.y)
        f_right = abs(obs.foot["foot_r"].ground.force.y)
        vertical_force = f_left + f_right

        mass = obs.norm_spec.mass  # type: ignore
        g = obs.norm_spec.g  # type: ignore
        weight = mass * g
        norm_force = vertical_force / weight

        reward = np.exp(self.sensitivity * (norm_force - 1.0) ** 2)

        return reward * self.scale

    def reset(self, model: opensim.Model, state: opensim.State, obs: Observation):
        pass


# class JointLoadPenalty(Reward):
#     """
#     Penalize joint reaction force magnitude (BW-normalized) and loading rate.
#     Uses Joint.calcReactionOnParentExpressedInGround(state).
#     """
#     def __init__(self, joints: Optional[Sequence[str]] = None, bw_norm: bool = True,
#                  scale_force: float = 0.05, scale_rate: float = 1e-3):
#         self.joint_names = list(joints) if joints else []
#         self.bw_norm = bw_norm
#         self.sf = scale_force
#         self.sr = scale_rate
#         self._prev_F: Dict[str, float] = {}
#
#     def reset(self, model: opensim.Model, state: opensim.State, obs: Observation):
#         self._prev_F.clear()
#
#     def compute(self, model: opensim.Model, state: opensim.State, obs: Observation, act: Action) -> float:
#         if not self.joint_names:
#             # default: penalize all joints in the model
#             self.joint_names = [model.getJointSet().get(i).getName()
#                                 for i in range(model.getJointSet().getSize())]
#         Mg = model.getTotalMass(state) * abs(model.get_gravity()[1]) if self.bw_norm else 1.0
#
#         dt = float(getattr(obs, "dt", 0.01.0))
#         cost = 0.0
#         for name in self.joint_names:
#             j = model.getJointSet().get(name)
#             _, F = _joint_spatial_force_G(j, state)  # N
#             Fn = float(np.linalg.norm(F))
#             prev = self._prev_F.get(name, Fn)
#             dF = (Fn - prev) / max(dt, 1e-6)
#             cost += self.sf * (Fn / max(Mg, 1e-6)) + self.sr * abs(dF) / max(Mg, 1e-6)
#             self._prev_F[name] = Fn
#         return -cost

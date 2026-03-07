import opensim

from environment.osim.action import Action
from environment.osim.observation import Observation

from .base import RewardComponent


class HeadStabilityReward(RewardComponent):
    __slots__ = ["acc_scale", "ang_vel_scale", "head_marker_name"]

    def __init__(
        self,
        head_marker_name: str = "Top.Head",
        acc_scale: float = 1.0,
        ang_vel_scale: float = 1.0,
    ):
        self.head_marker_name = head_marker_name
        self.acc_scale = acc_scale
        self.ang_vel_scale = ang_vel_scale

    def compute(
        self,
        model: opensim.Model,
        state: opensim.State,
        obs: Observation,
        action: Action,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """
        Head stability = -(acc_scale * linear acceleration^2 + ang_vel_scale * angular velocity^2)
        However computing only linear acceleration as of right now
        """
        head_marker = obs.marker[self.head_marker_name]
        acc = head_marker.acc
        acc_cost = -self.acc_scale * (acc.x**2 + acc.y**2 + acc.z**2)

        return acc_cost


class UprightReward(RewardComponent):
    __slots__ = ["scale"]

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def compute(
        self,
        model,
        state,
        obs: Observation,
        action,
        terminated,
        truncated,
    ) -> float:
        pitch = obs.pelvis.ang.x
        roll = obs.pelvis.ang.z

        penalty = pitch**2 + roll**2

        return -penalty * self.scale


# class KneeBendReward(RewardComponent):
#     __slots__ = ["scale", "target_angle", "sigma", "knee_names"]

#     def __init__(
#         self,
#         scale: float = 1.0,
#         target_angle: float = -0.5,  # approx -28 degrees (flexion is usually negative in OpenSim)
#         sigma: float = 0.5,
#         knee_names: tuple = ("knee_r", "knee_l"),
#     ):
#         self.scale = scale
#         self.target_angle = target_angle
#         self.sigma = sigma
#         self.knee_names = knee_names

#     def compute(
#         self,
#         model: opensim.Model,
#         state: opensim.State,
#         obs: Observation,
#         action: Action,
#         terminated: bool,
#         truncated: bool,
#     ) -> float:
#         total_reward = 0.0

#         # Access joint coordinates directly from model/state for safety
#         # (Coordinate names in OpenSim standard models are usually 'knee_angle_r' or 'knee_r')
#         coords = model.getCoordinateSet()

#         for name in self.knee_names:
#             # Try specific suffix first (L2M models often use 'knee_angle_r')
#             # Adjust string lookup based on your specific .osim file
#             try:
#                 coord = coords.get(f"{name}_angle")
#             except:
#                 coord = coords.get(name)

#             if not coord:
#                 continue

#             # Get angle in radians
#             angle = coord.getValue(state)

#             # Gaussian reward: exp( - (angle - target)^2 / (2 * sigma^2) )
#             # 1.0 when at target, decays to 0.0 as it moves away
#             diff = angle - self.target_angle
#             reward = math.exp(-(diff**2) / (2 * self.sigma**2))
#             total_reward += reward

#         return total_reward * self.scale

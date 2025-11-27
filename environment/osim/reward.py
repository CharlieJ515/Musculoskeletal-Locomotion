from abc import ABC, abstractmethod
from typing import Dict, Tuple
import dataclasses

import opensim

from .observation import FootState, Observation
from .action import Action


class RewardComponent(ABC):
    @abstractmethod
    def compute(
        self,
        model: opensim.Model,
        state: opensim.State,
        obs: Observation,
        action: Action,
        terminated: bool,
        truncated: bool,
    ) -> float:
        pass

    def reset(self, model: opensim.Model, state: opensim.State, obs: Observation):
        pass


#
# def _require_stage(model: osim.Model, state: osim.State, stage: str = "Velocity"):
#     """
#     Ensure the State is realized to at least the requested stage.
#     Valid: "Position", "Velocity", "Dynamics", "Acceleration", "Report"
#     """
#     stages = {
#         "Position": osim.Stage.Position,
#         "Velocity": osim.Stage.Velocity,
#         "Dynamics": osim.Stage.Dynamics,
#         "Acceleration": osim.Stage.Acceleration,
#         "Report": osim.Stage.Report,
#     }
#     model.realize(state, stages[stage])
#
#
# def _joint_spatial_force_G(joint: osim.Joint, state: osim.State):
#     """
#     Reaction on parent, expressed in Ground. Returns (moment_G, force_G) as np arrays.
#     """
#     # requires Acceleration stage
#     _require_stage(joint.getModel(), state, "Acceleration")
#     sv = joint.calcReactionOnParentExpressedInGround(state)  # SimTK::SpatialVec {moment; force}
#     M = np.array([sv[0][0], sv[0][1], sv[0][2]])
#     F = np.array([sv[1][0], sv[1][1], sv[1][2]])
#     return M, F
#
#
# def _muscle_positive_mech_power(m: osim.Muscle, state: osim.State) -> float:
#     """
#     Mechanical power proxy (W): max(F_fiber * v_fiber, 0).
#     """
#     _require_stage(m.getModel(), state, "Dynamics")
#     F = m.getFiberForce(state)     # N
#     v = m.getFiberVelocity(state)  # m/s (sign convention per model)
#     P = F * v
#     return float(max(P, 0.0))
#


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


class EnergyReward(RewardComponent):
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
        energy = 0.0
        for name, muscle in obs.muscle.items():
            energy += muscle.activation**2
        energy = energy / len(obs.muscle)
        return -energy * self.scale


class SmoothnessReward(RewardComponent):
    __slots__ = ["_prev_action", "scale"]

    def __init__(self, scale: float = 1.0):
        self.scale = scale

    def reset(self, model: opensim.Model, state: opensim.State, obs: Observation):
        self._prev_action = Action.from_opensim(model, state)

    def compute(
        self,
        model: opensim.Model,
        state: opensim.State,
        obs: Observation,
        action: Action,
        terminated: bool,
        truncated: bool,
    ) -> float:
        cost = 0
        for name in Action.muscle_order:
            current_act = action[name]
            prev_act = self._prev_action[name]

            cost += (current_act - prev_act) ** 2
        cost = cost / len(Action.muscle_order)
        self._prev_action = action
        return -cost * self.scale


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


class CompositeReward:
    __slots__ = ["components", "weights"]

    def __init__(
        self, components: Dict[str, RewardComponent], weights: Dict[str, float]
    ):
        self.components = components
        self.weights = weights

        self._check_key()

    def _check_key(self):
        component_keys = set(self.components.keys())
        weight_keys = set(self.weights.keys())

        missing_weights = component_keys - weight_keys
        missing_components = weight_keys - component_keys

        if missing_weights or missing_components:
            parts = []
            if missing_weights:
                parts.append(f"weights missing keys: {sorted(missing_weights)}")
            if missing_components:
                parts.append(f"components missing keys: {sorted(missing_components)}")
            raise ValueError("Reward mapping key mismatch: " + "; ".join(parts))

    def reset(self, model: opensim.Model, state: opensim.State, obs: Observation):
        for t in self.components.values():
            t.reset(model, state, obs)

    def compute(
        self,
        model: opensim.Model,
        state: opensim.State,
        obs: Observation,
        action: Action,
        terminated: bool,
        truncated: bool,
    ) -> Tuple[float, Dict[str, float]]:
        total = 0.0
        rewards: Dict[str, float] = {}
        for name, component in self.components.items():
            w = self.weights[name]
            reward = component.compute(model, state, obs, action, terminated, truncated)
            total += w * reward
            rewards[name] = reward
        return float(total), rewards


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

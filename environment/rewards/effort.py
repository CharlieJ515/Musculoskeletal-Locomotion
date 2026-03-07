import opensim

from environment.osim.action import Action
from environment.osim.observation import Observation

from .base import RewardComponent


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

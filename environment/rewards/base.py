from abc import ABC, abstractmethod
from typing import Dict, Tuple

import opensim

from environment.osim.action import Action
from environment.osim.observation import Observation


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

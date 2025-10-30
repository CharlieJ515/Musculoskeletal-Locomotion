from pathlib import Path
from typing import Any, Dict, Tuple, TypeVar

import opensim
import gymnasium as gym

from .osim_model import OsimModel
from .observation import Observation
from .action import Action
from .reward import CompositeReward
from .pose import Pose


class OsimEnv(gym.Env[Observation, Action]):
    def __init__(
        self,
        model_path: Path,
        init_pose: Pose,
        *,
        visualize: bool = True,
        integrator_accuracy: float = 5e-5,
        stepsize: float = 0.01,
        time_limit: int = 60,
    ):
        self.osim_model = OsimModel(
            model_path,
            visualize,
            integrator_accuracy,
            stepsize,
        )
        self.init_pose = init_pose

        self.visualize = visualize
        self.integrator_accuracy = integrator_accuracy
        self.time_limit = time_limit

        self.total_step = 0

    def _get_obs(self) -> Observation:
        return self.osim_model.get_obs()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[Observation, Dict[str, Any]]:
        super().reset(seed=seed)

        self.osim_model.reset(self.init_pose)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(
        self, action: Action
    ) -> Tuple[Observation, float, bool, bool, Dict[str, Any]]:
        self.osim_model.actuate(action)
        self.osim_model.integrate()

        obs = self._get_obs()
        reward = 0

        info: Dict[str, Any] = {}
        terminated, truncated = False, False
        if obs.pelvis.pos.y < 0.6:
            terminated = True
            info["terminated_reason"] = "pelvis_height_drop"

        if self.osim_model.step * self.osim_model.stepsize > self.time_limit:
            truncated = True
            info["truncated_reason"] = "time_limit_exceeded"

        return obs, reward, terminated, truncated, info

    def render(self):
        raise NotImplementedError(
            "Render method is not supported for this environment."
        )

    @property
    def model(self) -> opensim.Model:
        return self.osim_model.model

    @property
    def state(self) -> opensim.State:
        return self.osim_model.state

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
        terminated = False
        truncated, trunc_reason = self._get_truncated(obs)

        info: Dict[str, Any] = {}
        if trunc_reason is not None:
            info["truncated_reason"] = trunc_reason

        return obs, reward, terminated, truncated, info

    def _get_truncated(self, obs: Observation) -> Tuple[bool, str | None]:
        if self.osim_model.step * self.osim_model.stepsize > self.time_limit:
            return True, "time_limit_exceeded"

        if obs.body["pelvis"].pos.y < 0.6:
            return True, "pelvis_height_drop"

        return False, None

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

from typing import TypeVar, Any, SupportsFloat
import math

import gymnasium as gym
import opensim

from environment.osim import OsimEnv, Observation
from environment.osim.index import coordinate_index

ActType = TypeVar("ActType")


class BabyStepsWrapper(gym.Wrapper[Observation, ActType, Observation, ActType]):
    def __init__(
        self,
        env: gym.Env[Observation, ActType],
        *,
        height_name: str = "pelvis_ty",
        tilt_name: str = "pelvis_tilt",
        height_limit: float = 0.75,
        tilt_limit: tuple[float, float] = (math.radians(-15), math.radians(7)),
    ):
        super().__init__(env)
        self.height_name = height_name
        self.tilt_name = tilt_name

        self.height_limit = height_limit
        self.tilt_limit = tilt_limit

    def _get_model(self) -> opensim.Model:
        base_env: OsimEnv = self.unwrapped  # type: ignore[reportAssignmentType]
        return base_env.model

    def _get_state(self) -> opensim.State:
        base_env: OsimEnv = self.unwrapped  # type: ignore[reportAssignmentType]
        return base_env.state

    def _get_obs(self) -> Observation:
        base_env: OsimEnv = self.unwrapped  # type: ignore[reportAssignmentType]
        return base_env._get_obs()

    def _assist(self, obs: Observation) -> Observation:
        model = self._get_model()
        state = self._get_state()
        coord_set: opensim.CoordinateSet = model.getCoordinateSet()

        updated = False

        height_idx = coordinate_index(self.height_name)
        height_coord: opensim.Coordinate = coord_set.get(height_idx)
        current_height = height_coord.getValue(state)
        if current_height < self.height_limit:
            height_coord.setValue(state, self.height_limit)
            updated = True

        tilt_idx = coordinate_index(self.tilt_name)
        tilt_coord: opensim.Coordinate = coord_set.get(tilt_idx)
        current_tilt = tilt_coord.getValue(state)
        if current_tilt < self.tilt_limit[0]:
            tilt_coord.setValue(state, self.tilt_limit[0])
            updated = True
        elif current_tilt > self.tilt_limit[1]:
            tilt_coord.setValue(state, self.tilt_limit[1])
            updated = True

        if not updated:
            return obs

        model.realizeDynamics(state)
        obs = self._get_obs()
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        obs = self._assist(obs)

        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[Observation, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._assist(obs)

        return obs, reward, terminated, truncated, info

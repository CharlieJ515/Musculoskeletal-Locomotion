from typing import Optional, TypeVar, Any, SupportsFloat
from pathlib import Path
import warnings

import gymnasium as gym
import opensim

from environment.osim import OsimEnv

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class MotionLoggerWrapper(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    def __init__(
        self,
        env: gym.Env[ObsType, ActType],
        output_dir: Path = Path("./output"),
        filename_format: str = "run_{:04d}.mot",
    ):
        super().__init__(env)

        self.env = env
        self.filename_format = filename_format
        self.output_dir = output_dir
        self.episode_idx = 0

        if not self.filename_format.endswith(".mot"):
            raise ValueError(
                f"file name must end with .mot, got {self.filename_format}"
            )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        model = self._get_model()
        coord_set = model.getCoordinateSet()
        self.coord_names = [
            coord_set.get(i).getName() for i in range(coord_set.getSize())
        ]

        self.table: Optional[opensim.TimeSeriesTable] = None

    def _get_model(self) -> opensim.Model:
        base_env: OsimEnv = self.unwrapped  # type: ignore[reportAssignmentType]
        return base_env.model

    def _get_state(self) -> opensim.State:
        base_env: OsimEnv = self.unwrapped  # type: ignore[reportAssignmentType]
        return base_env.state

    def _reset_table(self, coord_names: list[str]):
        if self.table is not None:
            raise RuntimeError(
                f"Attempting to create new table before saving existing one."
            )

        self.table = opensim.TimeSeriesTable()
        self.table.setColumnLabels(coord_names)
        self.table.addTableMetaDataString("inDegrees", "no")

    def _sample_coordinates(
        self,
        model: opensim.Model,
        state: opensim.State,
    ) -> list[float]:
        model.realizePosition(state)

        coord_set: opensim.CoordinateSet = model.getCoordinateSet()
        coord_size = coord_set.getSize()
        vals = [0.0] * coord_size
        for i in range(coord_size):
            coord: opensim.Coordinate = coord_set.get(i)
            v = coord.getValue(state)
            vals[i] = v

        return vals

    def _append_row(self, t: float, values: list[float]):
        if len(values) != len(self.coord_names):
            raise ValueError()
        row = opensim.RowVector(values)
        self.table.appendRow(t, row)  # type: ignore[reportOptionalMemberAccess]

    def _episode_filename(self, idx: int) -> str:
        return self.filename_format.format(idx)

    def _write_file(self) -> Path:
        filename = self._episode_filename(self.episode_idx)
        path = self.output_dir / filename
        opensim.STOFileAdapter.write(self.table, str(path))

        self.table = None
        return path

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        obs = self.env.reset(seed=seed, options=options)

        self._reset_table(self.coord_names)

        model = self._get_model()
        state = self._get_state()
        t = state.getTime()

        values = self._sample_coordinates(model, state)
        self._append_row(t, values)
        return obs

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        out = self.env.step(action)
        _, _, terminated, truncated, info = out

        model = self._get_model()
        state = self._get_state()
        t = state.getTime()

        values = self._sample_coordinates(model, state)
        self._append_row(t, values)

        if terminated or truncated:
            path = self._write_file()
            self.episode_idx += 1
            info["mot_path"] = path

        return out

    def close(self):
        if self.table is not None:
            path = self._write_file()
            warnings.warn(
                f"MotionLoggerWrapper: unflushed data written to {path}", RuntimeWarning
            )

        return self.env.close()

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Optional, cast

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from osim.env import L2M2019Env as _L2M2019Env


class L2M2019GymEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {}

    def __init__(
        self,
        model_path: Path,
        difficulty: int = 0,
        seed: Optional[int] = None,
        visualize: bool = False,
    ) -> None:
        self.difficulty = difficulty
        self.env = _L2M2019Env(visualize=visualize, seed=seed, difficulty=self.difficulty)

        if not model_path.exists():
            raise FileNotFoundError(f"OpenSim model not found: {model_path}")
        if model_path.suffix.lower() != ".osim":
            raise ValueError(f"Expected an .osim file, got: {model_path.suffix} (path={model_path})")
        self._model_path = model_path
        self.env.load_model(str(self._model_path))

        self.observation_space = cast(Box, self.env.observation_space)
        self.action_space = cast(Box, self.env.action_space)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        obs = self.env.reset(seed=seed, project=True, obs_as_dict=False)
        obs = np.array(obs, dtype=np.float32)
        info: Dict[str, Any] = {}

        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        obs, reward, truncated, info = self.env.step(action, project=True, obs_as_dict=False)
        obs = np.array(obs, dtype=np.float32)

        return obs, reward, False, truncated, info

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        raise NotImplementedError()

    def close(self) -> None:
        return


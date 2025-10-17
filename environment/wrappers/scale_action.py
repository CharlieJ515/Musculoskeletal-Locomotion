from __future__ import annotations

import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import ActionWrapper


class ScaleAction(ActionWrapper):
    def __init__(self, env: gym.Env, source_space: Box):
        super().__init__(env)

        assert isinstance(env.action_space, Box), "Underlying env must use Box action space."
        assert source_space.shape == env.action_space.shape, (
            f"Shape mismatch: source {source_space.shape} vs env {env.action_space.shape}"
        )
        assert np.issubdtype(env.action_space.dtype, np.floating), "Action dtype must be float."
        assert np.issubdtype(source_space.dtype, np.floating), "source_space dtype must be float."
        assert np.dtype(source_space.dtype) == np.dtype(env.action_space.dtype), (
            f"dtype mismatch: source {source_space.dtype} vs env {env.action_space.dtype}"
        )

        _dtype = source_space.dtype
        s_low = np.asarray(source_space.low, dtype=_dtype)
        s_high = np.asarray(source_space.high, dtype=_dtype)
        t_low = np.asarray(env.action_space.low, dtype=_dtype)
        t_high = np.asarray(env.action_space.high, dtype=_dtype)

        if not (np.isfinite(s_low).all() and np.isfinite(s_high).all()):
            raise AssertionError("source_space bounds must be finite.")
        denom = s_high - s_low
        if np.any(denom == 0.0):
            raise AssertionError("source_space has zero-width bounds in at least one dimension.")

        self._scale = (t_high - t_low) / denom
        self._bias = t_low - s_low * self._scale

        self.action_space = Box(
            low=source_space.low,
            high=source_space.high,
            shape=source_space.shape,
            dtype=_dtype.type,
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=self.action_space.dtype)
        return a * self._scale + self._bias

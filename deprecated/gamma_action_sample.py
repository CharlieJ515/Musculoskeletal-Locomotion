import gymnasium as gym
import numpy as np


def random_action_gamma(
    action_space: gym.spaces.Box,
    shape: float = 0.8,
    scale: float = 1.0,
    clip_percentiles: tuple[float, float] = (0.5, 99.5),
) -> np.ndarray:
    # Sample from Gamma(k, θ)
    samples = np.random.gamma(shape=shape, scale=scale, size=action_space.shape)

    # Normalize to (-1, 1)
    low_p, high_p = np.percentile(samples, clip_percentiles)
    samples = np.clip(samples, low_p, high_p)
    samples_norm = 2 * (samples - low_p) / (high_p - low_p) - 1

    # Map to actual env action space
    action_low, action_high = action_space.low, action_space.high
    action = action_low + (samples_norm + 1) * 0.5 * (action_high - action_low)

    return np.clip(action, action_low, action_high)

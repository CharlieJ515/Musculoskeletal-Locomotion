from typing import Any

import torch.nn as nn

from configs import NoiseConfig, PERConfig, ReplayBufferConfig, SACConfig, TD3Config
from rl import (
    PER,
    SAC,
    TD3,
    BaseNoise,
    BaseReplayBuffer,
    BaseRL,
    GaussianNoise,
    OUNoise,
    ReplayBuffer,
)

AGENT_REGISTRY = {
    "TD3": (TD3, TD3Config),
    "SAC": (SAC, SACConfig),
}

BUFFER_REGISTRY = {
    "PER": (PER, PERConfig),
    "ReplayBuffer": (ReplayBuffer, ReplayBufferConfig),
}

NOISE_REGISTRY: dict[str, type[BaseNoise]] = {
    "GaussianNoise": GaussianNoise,
    "OUNoise": OUNoise,
}


def create_agent(
    agent_cfg_dict: dict[str, Any],
    actor_net: type[nn.Module],
    critic_net: type[nn.Module],
) -> BaseRL:
    cfg = agent_cfg_dict.copy()

    agent_type = cfg.pop("type", None)
    if agent_type is None:
        raise ValueError("Agent config must include a 'type' key (e.g., type: TD3)")
    if agent_type not in AGENT_REGISTRY:
        raise ValueError(f"Unknown agent type '{agent_type}'")

    agent_cls, config_cls = AGENT_REGISTRY[agent_type]
    agent_config = config_cls.from_dict(cfg, actor_net=actor_net, critic_net=critic_net)
    agent = agent_cls.from_config(agent_config)

    return agent


def create_buffer(buffer_cfg_dict: dict[str, Any]) -> BaseReplayBuffer:
    cfg = buffer_cfg_dict.copy()

    buffer_type = cfg.pop("type", None)
    if buffer_type is None:
        raise ValueError("Buffer config must include a 'type' key (e.g., type: PER)")
    if buffer_type not in BUFFER_REGISTRY:
        raise ValueError(f"Unknown buffer type '{buffer_type}'")

    buffer_cls, config_cls = BUFFER_REGISTRY[buffer_type]
    buffer_config = config_cls.from_dict(cfg)
    buffer = buffer_cls.from_config(buffer_config)

    return buffer


def create_noise_sampler(
    cfg: NoiseConfig, batch_size: int, action_dim: tuple[int, ...]
) -> BaseNoise:
    if cfg.name not in NOISE_REGISTRY:
        raise ValueError(f"Unknown noise generator: {cfg.name}")

    NoiseClass = NOISE_REGISTRY[cfg.name]

    kwargs = {
        "batch_size": batch_size,
        "action_dim": action_dim,
        "sigma_start": cfg.sigma,
        "sigma_min": cfg.sigma_min,
        "decay_steps": cfg.decay_steps,
    }

    if cfg.name == "OUNoise":
        kwargs["theta"] = cfg.theta
        kwargs["dt"] = cfg.dt
    elif cfg.name == "GaussianNoise":
        kwargs["clip"] = cfg.clip

    return NoiseClass(**kwargs)

from typing import cast

import gymnasium as gym

from configs import TrainConfig
from environment.osim import OsimEnv
from environment.rewards import REWARD_REGISTRY, CompositeReward
from environment.wrappers import WRAPPER_REGISTRY, LimitForceConfig
from environment.wrappers.composite_reward import CompositeRewardWrapper


def build_composite_reward(reward_configs: list[dict]) -> CompositeReward | None:
    if not reward_configs:
        return None

    reward_components = {}
    reward_weights = {}

    for r_cfg in reward_configs:
        r_name = r_cfg["name"]
        r_key = r_cfg.get("key", r_name)
        r_params = r_cfg.get("params", {}).copy()
        r_weight = r_cfg.get("weight", 1.0)

        if r_name not in REWARD_REGISTRY:
            raise ValueError(f"Unknown reward component: {r_name}")

        RewardClass = REWARD_REGISTRY[r_name]
        reward_components[r_key] = RewardClass(**r_params)
        reward_weights[r_key] = float(r_weight)

    return CompositeReward(reward_components, reward_weights)


def create_env(cfg: TrainConfig, visualize: bool = False) -> gym.Env:
    env = OsimEnv(cfg.model, cfg.pose, visualize=visualize)

    for w_cfg in cfg.wrappers:
        w_name = w_cfg["name"]
        w_params = w_cfg.get("params", {}).copy()

        if w_name not in WRAPPER_REGISTRY:
            raise ValueError(f"Unknown wrapper: {w_name}")

        if w_name == "CompositeRewardWrapper":

            reward_fn = build_composite_reward(cfg.rewards)
            if reward_fn is None:
                raise ValueError(
                    "Failed to build CompositeReward. "
                    "Check 'rewards' in the configuration."
                )

            WrapperClass = cast(type[CompositeRewardWrapper], WRAPPER_REGISTRY[w_name])
            env = WrapperClass(env, reward_fn)
            continue

        if w_name == "BabyWalkerWrapper" and "limit_configs" in w_params:
            configs = []
            for lc in w_params["limit_configs"]:
                configs.append(LimitForceConfig(**lc))
            w_params["limit_configs"] = configs

        WrapperClass = WRAPPER_REGISTRY[w_name]
        env = WrapperClass(env, **w_params)

    return env

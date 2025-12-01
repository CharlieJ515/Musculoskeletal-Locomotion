from typing import cast, Literal
from pathlib import Path
from dataclasses import dataclass

import mlflow
import torch
import numpy as np
import gymnasium as gym
from tqdm import tqdm
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from environment.models import gait14dof22_path
from environment.osim import OsimEnv
from environment.osim.pose import Pose, get_default_pose
from environment.osim.reward import (
    CompositeReward,
    AliveReward,
    VelocityReward,
    EnergyReward,
    UprightReward,
    FootstepReward,
)
from environment.wrappers import (
    TargetSpeedWrapper,
    MotionLoggerWrapper,
    SimpleEnvWrapper,
    CompositeRewardWrapper,
    FrameSkipWrapper,
    BabyWalkerWrapper,
    LimitForceConfig,
)
from rl.td3 import TD3, TD3Config
from rl.replay_buffer.prioritized_replay_buffer import PER, PERConfig
from rl.transition import TransitionBatch
from models.td3_mlp import MLPActor, MLPCritic

from analysis.tensorboard_utils.writer import get_writer
from analysis.mlflow_utils.start import start_mlflow
from analysis.tensorboard_utils.distribution import log_rewards, log_preds

from utils.tmp_dir import get_tmp, clear_tmp
from utils.save import save_ckpt
from utils.exploration import OUNoise
from deprecated.gamma_action_sample import random_action_gamma


@dataclass
class TrainConfig:
    total_steps: int
    start_random: int
    batch_size: int
    eval_interval: int
    eval_episodes: int
    log_interval: int
    seed: int

    model: Path
    pose: Pose
    visualize: bool
    num_env: int
    reward_key: list[str]
    mp_context: Literal["spawn", "fork", "forkserver"]

    def log_params(self):
        mlflow.log_params(
            {
                "total_steps": self.total_steps,
                "start_random": self.start_random,
                "batch_size": self.batch_size,
                "eval_interval": self.eval_interval,
                "eval_episodes": self.eval_episodes,
                "log_interval": self.log_interval,
                "seed": self.seed,
                "init_pose": self.pose.name,
                "opensim_model": str(self.model),
                "visualize": self.visualize,
                "num_env": self.num_env,
                "reward_key": self.reward_key,
                "mp_context": self.mp_context,
            }
        )

    def __post_init__(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)


@torch.no_grad()
def evaluate(
    model: Path,
    pose: Pose,
    agent: TD3,
    step: int,
    episodes: int,
    reward_key: list[str],
):
    active_run = mlflow.active_run()
    active_run_name = active_run.data.tags.get("mlflow.runName")  # type: ignore[reportOptionalMemberAccess]

    tmpdir = get_tmp()
    returns = []

    motion_output_dir = tmpdir / f"{active_run_name}_eval_step{step}"
    env = create_env(model, pose, False, False)
    env = MotionLoggerWrapper(
        env,
        motion_output_dir,
        f"{active_run_name}_eval_step{step}_{{}}.mot",
    )
    agent.eval()
    for ep in range(episodes):
        run_name = f"{active_run_name}_eval_step{step}_{ep}"
        eval_dir = tmpdir / run_name
        eval_dir.mkdir(parents=True, exist_ok=True)

        mlflow.start_run(run_name=run_name, nested=True)
        mlflow.set_tags({"phase": "eval", "checkpoint_step": step, "episode": ep})

        all_actions = []
        ep_ret = 0.0
        ep_disc = 0.0
        disc_pow = 1.0

        s_np, info = env.reset()
        ep_step = 0
        while True:
            s_t = torch.tensor(
                s_np, dtype=torch.float32, device=agent.device
            ).unsqueeze(0)

            a_t = agent.select_action(s_t)
            a_np = a_t.squeeze(0).cpu().numpy()
            s_next_np, r, terminated, truncated, info = env.step(a_np)
            ep_ret += float(r)

            ep_disc += disc_pow * float(r)
            disc_pow *= float(agent.gamma)

            q1 = agent.Q1(s_t, a_t).detach().cpu().squeeze(0)
            q2 = agent.Q2(s_t, a_t).detach().cpu().squeeze(0)
            qmin = torch.min(q1, q2)

            metrics = {
                "cumulative_return": float(ep_ret),
                "discounted_return": float(ep_disc),
                **info["rewards"],
            }
            for i, key in enumerate(reward_key):
                metrics[f"q1_pred/{key}"] = q1[i]
                metrics[f"q2_pred/{key}"] = q2[i]
                metrics[f"min/{key}"] = qmin[i]
            mlflow.log_metrics(metrics, step=ep_step)
            all_actions.append(a_np)

            s_np = s_next_np
            ep_step += 1

            if terminated or truncated:
                mot_path = info["mot_path"]
                mlflow.log_artifact(str(mot_path), artifact_path="motion")
                break

        # log total returns
        mlflow.log_metrics(
            {
                "total_return": float(ep_ret),
                "total_discounted_return": float(ep_disc),
                "episode_length": int(ep_step),
            },
            step=ep_step,
        )

        # plot and log action distribution
        A = np.asarray(all_actions)
        A_flat = A.reshape(-1)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(A_flat, bins=100, log=True)
        ax.set_title(f"Action distribution step {step}, eval ep {ep}")
        ax.set_xlabel("action value")
        ax.set_ylabel("count (log)")
        mlflow.log_figure(fig, f"figures/action_hist_step{step}_ep{ep}.png")
        plt.close(fig)

        returns.append(ep_ret)
        mlflow.end_run()

    env.close()
    agent.train(True)
    return float(np.mean(returns))


def create_env(
    model: Path,
    pose: Pose,
    visualize: bool,
    baby_walker: bool,
) -> gym.Env:
    osim_env = OsimEnv(model, pose, visualize=visualize)
    env = gym.wrappers.TimeLimit(osim_env, 1000)
    env = TargetSpeedWrapper(env, speed_range=(0.25, 0.75))

    if baby_walker:
        env = BabyWalkerWrapper(
            env,
            [
                LimitForceConfig(
                    coordinate_name="pelvis_ty",
                    upper_limit=1.5,
                    upper_stiffness=100.0,
                    lower_limit=0.75,
                    lower_stiffness=2000.0,
                    damping=50.0,
                    transition=0.1,
                    dissipate_energy=True,
                ),
                LimitForceConfig(
                    coordinate_name="pelvis_tilt",
                    upper_limit=15.0,
                    upper_stiffness=500.0,
                    lower_limit=-15.0,
                    lower_stiffness=500.0,
                    damping=10.0,
                    transition=0.1,
                    dissipate_energy=True,
                ),
                LimitForceConfig(
                    coordinate_name="pelvis_list",
                    upper_limit=15.0,
                    upper_stiffness=500.0,
                    lower_limit=-15.0,
                    lower_stiffness=500.0,
                    damping=10.0,
                    transition=0.1,
                    dissipate_energy=True,
                ),
            ],
            decay_steps=30_000 * 4,
        )

    reward_components = {
        "alive_reward": AliveReward(0.1, -200),
        "velocity_reward": VelocityReward(1.0),
        "energy_reward": EnergyReward(1.0),
        "footstep_reward": FootstepReward(5.0, stepsize=osim_env.osim_model.stepsize),
        "upright_reward": UprightReward(1.0),
    }
    reward_weights = {
        "alive_reward": 1.0,
        "velocity_reward": 1.0,
        "energy_reward": 1.0,
        "footstep_reward": 1.0,
        "upright_reward": 1.0,
    }
    reward_fn = CompositeReward(reward_components, reward_weights)
    env = CompositeRewardWrapper(env, reward_fn)
    env = FrameSkipWrapper(env, 4)
    env = SimpleEnvWrapper(env)
    env = gym.wrappers.TimeAwareObservation(env, flatten=True, normalize_time=True)
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)
    env = gym.wrappers.RescaleAction(env, np.float32(-1.0), np.float32(1.0))
    return env


def reward_info_to_ndarray(
    reward_key: list[str], reward_info: dict[str, np.ndarray]
) -> np.ndarray:
    reward = np.array(
        [reward_info[key] for key in reward_key],
        dtype=np.float32,
    ).T
    return reward


def main(
    cfg: TrainConfig,
    td3_cfg: TD3Config,
    per_cfg: PERConfig,
):
    env = gym.vector.AsyncVectorEnv(
        [
            lambda: create_env(cfg.model, cfg.pose, cfg.visualize, True)
            for i in range(cfg.num_env)
        ],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
        context=cfg.mp_context,
    )
    # create writer after creating child processes
    writer = get_writer()

    # TD3 hyperparams
    agent = TD3.from_config(td3_cfg)
    agent.log_params(prefix="agent/")

    # Replay buffer
    rb = PER.from_config(per_cfg)
    rb.log_params(prefix="buffer/")

    # Random Exploration
    noise_sampler = OUNoise(cfg.num_env, act_shape, sigma=0.1)
    s_np, infos = env.reset(seed=cfg.seed)
    episode_start = np.array([False] * env.num_envs, np.bool)
    print("Starting random action exploration")
    for t in range(1, cfg.start_random):
        a_np = noise_sampler.sample().clip(-1.0, 1.0)
        s_next_np, r, terminated, truncated, info = env.step(a_np)

        if episode_start.all():
            s_np = s_next_np
            episode_start = np.logical_or(terminated, truncated)
            noise_sampler.reset()
            continue

        reward_info = info["rewards"]
        reward = reward_info_to_ndarray(cfg.reward_key, reward_info)

        # Store transition
        idx = ~episode_start
        tr = TransitionBatch(
            obs=torch.as_tensor(s_np[idx], dtype=torch.float32),
            actions=torch.as_tensor(a_np[idx], dtype=torch.float32),
            rewards=torch.as_tensor(reward[idx], dtype=torch.float32),
            next_obs=torch.as_tensor(s_next_np[idx], dtype=torch.float32),
            dones=torch.as_tensor(terminated[idx], dtype=torch.bool),
        )
        rb.add(tr)

        s_np = s_next_np
        episode_start = np.logical_or(terminated, truncated)
        noise_sampler.reset(episode_start)

    # Training
    agent.train()
    s_np, info = env.reset(seed=cfg.seed)
    episode_start = np.array([False] * env.num_envs, np.bool)
    print("Starting training")
    for t in tqdm(range(1, cfg.total_steps + 1)):
        s_t = torch.as_tensor(s_np, dtype=torch.float32, device=agent.device)
        a_t = agent.select_action(s_t)
        a_np = a_t.cpu().numpy()
        noise = noise_sampler.sample()
        a_np = (a_np + noise).clip(-1.0, 1.0)  # TD3 config

        env.step_async(a_np)

        batch = rb.sample(cfg.batch_size, pin_memory=True).to(
            agent.device, non_blocking=True
        )
        metrics = agent.update(batch)
        td_error: torch.Tensor = metrics["critic"]["td_error"]
        rb.update_priorities(batch.indices, td_error)  # type: ignore

        s_next_np, r, terminated, truncated, info = env.step_wait()

        if episode_start.all():
            s_np = s_next_np
            episode_start = np.logical_or(terminated, truncated)
            noise_sampler.reset()
            continue

        reward_info = info["rewards"]
        reward = reward_info_to_ndarray(cfg.reward_key, reward_info)

        # Store transition
        idx = ~episode_start
        tr = TransitionBatch(
            obs=torch.as_tensor(s_np[idx], dtype=torch.float32),
            actions=torch.as_tensor(a_np[idx], dtype=torch.float32),
            rewards=torch.as_tensor(reward[idx], dtype=torch.float32),
            next_obs=torch.as_tensor(s_next_np[idx], dtype=torch.float32),
            dones=torch.as_tensor(terminated[idx], dtype=torch.bool),
        )
        rb.add(tr)

        s_np = s_next_np
        episode_start = np.logical_or(terminated, truncated)
        noise_sampler.reset(episode_start)

        # logging
        critic_metrics = metrics["critic"]
        writer.add_scalar("train/critic/critic_loss", critic_metrics["critic_loss"], t)
        log_rewards(reward_info, idx, t, "transit/reward")
        if "actor" in metrics:
            actor_metrics = metrics["actor"]
            writer.add_scalar("train/actor/actor_loss", actor_metrics["actor_loss"], t)

        if t % cfg.log_interval == 0:
            q1_pred = critic_metrics["q1_pred"]
            log_preds(cfg.reward_key, q1_pred, t, "train/critic/q1_pred")
            q2_pred = critic_metrics["q2_pred"]
            log_preds(cfg.reward_key, q2_pred, t, "train/critic/q2_pred")
            q_target = critic_metrics["q_target"]
            log_preds(cfg.reward_key, q_target, t, "train/critic/q_target")

            if "actor" in metrics:
                actor_metrics = metrics["actor"]
                writer.add_histogram("train/actor/q", actor_metrics["q"], t)

        # Periodic eval
        if t % cfg.eval_interval == 0:
            save_ckpt(agent, f"td3_osim_step{t}.pt")

            agent.eval()
            avg_ret = evaluate(
                cfg.model, cfg.pose, agent, t, cfg.eval_episodes, cfg.reward_key
            )
            print(f"[{t:6d}] eval_return={avg_ret:.2f}")
            agent.train()

    # Final eval
    final_ret = evaluate(
        cfg.model, cfg.pose, agent, -1, cfg.eval_episodes, cfg.reward_key
    )
    print(f"Final average return over {cfg.eval_episodes} episodes: {final_ret:.2f}")

    # Save checkpoint
    save_ckpt(agent, "td3_osim.pt")
    env.close()


if __name__ == "__main__":

    cfg = TrainConfig(
        total_steps=100_000,
        start_random=100,
        batch_size=256,
        eval_interval=5_000,
        eval_episodes=1,
        log_interval=20,
        seed=42,
        model=gait14dof22_path,
        pose=get_default_pose(),
        visualize=False,
        num_env=32,
        reward_key=[
            "alive_reward",
            "velocity_reward",
            "energy_reward",
            "footstep_reward",
            "upright_reward",
        ],
        mp_context="forkserver",
    )

    temp_env = create_env(cfg.model, cfg.pose, False, True)
    obs_shape = cast(tuple[int, ...], temp_env.observation_space.shape)
    act_shape = cast(tuple[int, ...], temp_env.action_space.shape)
    reward_shape = (len(cfg.reward_key),)
    temp_env.close()

    td3_cfg = TD3Config(
        state_dim=obs_shape,
        action_dim=act_shape,
        actor_net=MLPActor,
        critic_net=MLPCritic,
        reward_dim=reward_shape,
        reward_weight=torch.ones(len(cfg.reward_key)),
        policy_noise=0.1,
        noise_clip=0.25,
        policy_update_freq=2,
        lr=3e-4 / 4,
        load_chkpt=True,
        chkpt_file=Path("td3_osim.pt"),
    )
    per_cfg = PERConfig(
        capacity=100_000,
        obs_shape=obs_shape,
        action_shape=act_shape,
        reward_shape=reward_shape,
        alpha=0.6,
        beta_start=0.4,
        beta_frames=cfg.total_steps,
    )

    start_mlflow(
        "https://mlflow.kyusang-jang.com/capstone",
        "TD3-Osim",
        "td3_rb-size",
    )

    try:
        # main(cfg, td3_cfg, per_cfg)

        agent = TD3.from_config(td3_cfg)
        evaluate(cfg.model, cfg.pose, agent, 0, cfg.eval_episodes, cfg.reward_key)
    finally:
        mlflow.end_run()  # mlflow.end_run is idempotent
        clear_tmp()

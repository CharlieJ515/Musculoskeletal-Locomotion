from typing import cast
from pathlib import Path

import mlflow
import matplotlib.pyplot as plt
import torch
import numpy as np
import gymnasium as gym
import opensim

from environment.models import gait14dof22_path
from environment.osim import OsimEnv
from environment.osim.pose import Pose, get_bent_pose
from environment.osim.reward import (
    CompositeReward,
    AliveReward,
    VelocityReward,
    EnergyReward,
    SmoothnessReward,
    HeadStabilityReward,
    FootstepReward,
)
from environment.wrappers import (
    TargetSpeedWrapper,
    MotionLoggerWrapper,
    SimpleEnvWrapper,
    CompositeRewardWrapper,
    RescaleActionWrapper,
)
from rl.sac import SAC, default_target_entropy
from rl.replay_buffer.replay_buffer import ReplayBuffer
from rl.transition import TransitionBatch
from models.shallow_mlp import MLPActor, MLPCritic
from analysis.mlflow_utils.attempt import tag_attempt
from analysis.tensorboard_utils.writer import get_writer
from analysis.tensorboard_utils.distribution import (
    log_weight_hist,
    log_grad_hist,
    log_rewards,
    log_preds,
)
from utils.tmp_dir import get_tmp, clear_tmp
from utils.save import save_ckpt
from deprecated.gamma_action_sample import random_action_gamma


@torch.no_grad()
def evaluate(
    model: Path,
    pose: Pose,
    agent: SAC,
    step: int,
    episodes: int = 5,
):
    writer = get_writer()
    active_run = mlflow.active_run()
    active_run_name = active_run.data.tags.get("mlflow.runName")  # type: ignore[reportOptionalMemberAccess]

    gamma = agent.gamma
    tmpdir = get_tmp()
    returns = []

    motion_output_dir = tmpdir / f"{active_run_name}_eval_step{step}"
    train_env = create_env(model, pose)
    env = MotionLoggerWrapper(train_env, motion_output_dir)
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
            disc_pow *= float(gamma)

            q1 = agent.Q1(s_t, a_t).detach().cpu().squeeze(0)
            q2 = agent.Q2(s_t, a_t).detach().cpu().squeeze(0)
            qmin = torch.min(q1, q2)

            writer.add_scalars(
                f"eval/ep{ep}/critic/q1_pred",
                {
                    "alive": q1[0],
                    "velocity": q1[1],
                    "energy": q1[2],
                    "smoothness": q1[3],
                },
                ep_step,
            )
            writer.add_scalars(
                f"eval/ep{ep}/critic/q2_pred",
                {
                    "alive": q2[0],
                    "velocity": q2[1],
                    "energy": q2[2],
                    "smoothness": q2[3],
                },
                ep_step,
            )
            writer.add_scalars(
                f"eval/ep{ep}/critic/min",
                {
                    "alive": qmin[0],
                    "velocity": qmin[1],
                    "energy": qmin[2],
                    "smoothness": qmin[3],
                },
                ep_step,
            )
            writer.add_scalars(f"eval/ep{ep}/transit/reward", info["rewards"], ep_step)
            writer.add_scalars(
                f"eval/ep{ep}/transit/reward",
                {
                    "cumulative_return": float(ep_ret),
                    "discounted_return": float(ep_disc),
                },
                ep_step,
            )

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


def create_env(model: Path, pose: Pose) -> gym.Env:
    osim_env = OsimEnv(model, pose, visualize=True)
    time_limit_env = gym.wrappers.TimeLimit(osim_env, 500)
    target_env = TargetSpeedWrapper(time_limit_env)
    reward_components = {
        "alive_reward": AliveReward(0.1),
        "velocity_reward": VelocityReward(1.0),
        "energy_reward": EnergyReward(3.0),
        "smoothness_reward": SmoothnessReward(6.0),
        "head_stability_reward": HeadStabilityReward(acc_scale=0.01),
        "footstep_reward": FootstepReward(5.0, stepsize=osim_env.osim_model.stepsize),
    }
    reward_weights = {
        "alive_reward": 1.0,
        "velocity_reward": 1.0,
        "energy_reward": 1.0,
        "smoothness_reward": 1.0,
        "head_stability_reward": 1.0,
        "footstep_reward": 1.0,
    }
    reward_fn = CompositeReward(reward_components, reward_weights)
    reward_env = CompositeRewardWrapper(target_env, reward_fn)
    simple_env = SimpleEnvWrapper(reward_env)
    time_aware_env = gym.wrappers.TimeAwareObservation(
        simple_env, flatten=True, normalize_time=True
    )
    # rescale_env = RescaleActionWrapper(time_aware_env, "abs")
    rescale_env = gym.wrappers.RescaleAction(
        time_aware_env, np.float32(-1.0), np.float32(1.0)
    )
    return rescale_env


# def sample_gaussian_action(
#     action_space: gym.spaces.Box,
#     mean: float = 0.0,
#     std: float = 0.55,
# ) -> np.ndarray:
#     raw = np.random.normal(loc=mean, scale=std, size=action_space.shape)

#     return np.clip(raw, action_space.low, action_space.high)


def reward_info_to_ndarray(
    reward_key: list[str], reward_info: dict[str, np.ndarray]
) -> np.ndarray:
    reward = np.array(
        [reward_info[key] for key in reward_key],
        dtype=np.float32,
    ).T
    return reward


def main():
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_name = "SAC-Osim6"
    run_name = "simple sac"
    model = gait14dof22_path
    pose = get_bent_pose()

    writer = get_writer()
    mlflow.set_tracking_uri("https://mlflow.kyusang-jang.com/capstone")
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name)
    tag_attempt(experiment_name, run_name)

    mlflow.log_params(
        {
            "env_id": "OsimEnv",
            "opensim_model": str(model),
            "seed": seed,
            "gym_version": gym.__version__,
            "opensim_version": opensim.__version__,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        }
    )

    torch.manual_seed(seed)
    np.random.seed(seed)

    reward_key = [
        "alive_reward",
        "velocity_reward",
        "energy_reward",
        "smoothness_reward",
        "head_stability_reward",
        "footstep_reward",
    ]
    env = gym.vector.AsyncVectorEnv(
        [lambda: create_env(model, pose) for _ in range(2)],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )

    obs_shape = cast(tuple[int, ...], env.single_observation_space.shape)
    action_shape = cast(tuple[int, ...], env.single_action_space.shape)
    reward_shape = (len(reward_key),)
    reward_weights = torch.ones(len(reward_key), dtype=torch.float32)
    mlflow.log_params(
        {
            "obs_shape": obs_shape,
            "action_shape": action_shape,
            "reward_shape": reward_shape,
        }
    )

    # SAC hyperparams
    agent = SAC(
        gamma=0.9,
        state_dim=obs_shape,
        action_dim=action_shape,
        reward_dim=reward_shape,
        log_std_min=-20.0,
        log_std_max=2.0,
        actor_net=MLPActor,
        critic_net=MLPCritic,
        lr=3e-4,
        tau=0.005,
        # target_entropy=-22,  # std 1
        target_entropy=-16,  # std ~0.5
        # target_entropy=-4.7,  # std ~0.3
        weight_decay=0.0,
        policy_update_freq=1,
        reward_weight=reward_weights,
        device=device,
        use_jit=False,
        train=True,
    )
    agent.log_params(prefix="agent/")

    # Replay buffer
    rb = ReplayBuffer(
        capacity=10_000,
        obs_shape=obs_shape,
        action_shape=action_shape,
        reward_shape=reward_shape,
        device=torch.device("cpu"),
    )
    rb.log_params(prefix="buffer/")

    # Training loop
    total_steps = 100_000
    # total_steps = 1_000
    start_random = 1_000
    batch_size = 256
    eval_interval = 10_000
    mlflow.log_params(
        {
            "total_steps": total_steps,
            "start_random": start_random,
            "batch_size": batch_size,
            "eval_interval": eval_interval,
        }
    )

    s_np, infos = env.reset(seed=seed)
    episode_start = np.array([False] * env.num_envs, np.bool)
    print("Starting random action exploration")
    for t in range(1, start_random):
        a_np = random_action_gamma(
            env.action_space  # pyright: ignore[reportArgumentType]
        )
        s_next_np, r, terminated, truncated, info = env.step(a_np)

        if episode_start.all():
            s_np = s_next_np
            episode_start = np.logical_or(terminated, truncated)
            continue

        reward_info = info["rewards"]
        reward = reward_info_to_ndarray(reward_key, reward_info)

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

    agent.train()
    s_np, info = env.reset(seed=seed)
    episode_start = np.array([False] * env.num_envs, np.bool)
    print("Starting training")
    for t in range(1, total_steps + 1):
        s_t = torch.as_tensor(s_np, dtype=torch.float32, device=agent.device)
        a_t = agent.select_action(s_t)
        a_np = a_t.cpu().numpy()

        env.step_async(a_np)

        batch = rb.sample(batch_size, pin_memory=True).to(device, non_blocking=True)
        metrics = agent.update(batch)

        s_next_np, r, terminated, truncated, info = env.step_wait()

        if episode_start.all():
            s_np = s_next_np
            episode_start = np.logical_or(terminated, truncated)
            continue

        reward_info = info["rewards"]
        reward = reward_info_to_ndarray(reward_key, reward_info)

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

        alpha_metrics = metrics["alpha"]
        critic_metrics = metrics["critic"]
        actor_metrics = metrics["actor"]

        writer.add_scalars("train/alpha", alpha_metrics, t)
        writer.add_scalar("train/critic/critic_loss", critic_metrics["critic_loss"], t)
        writer.add_scalar("train/actor/actor_loss", actor_metrics["actor_loss"], t)
        log_rewards(reward_info, idx, t, "transit/reward")

        if t % 20 == 0:
            q1_pred = critic_metrics["q1_pred"]
            log_preds(reward_key, q1_pred, t, "train/critic/q1_pred")
            q2_pred = critic_metrics["q2_pred"]
            log_preds(reward_key, q2_pred, t, "train/critic/q2_pred")
            q_target = critic_metrics["q_target"]
            log_preds(reward_key, q_target, t, "train/critic/q_target")

            writer.add_histogram("train/actor/log_prob", actor_metrics["log_prob"], t)
            writer.add_histogram("train/actor/q", actor_metrics["q"], t)

        # writer.add_histogram("transit/obs", s_np, t)
        # writer.add_histogram("transit/action", a_np, t)

        # log_weight_hist(f"model/actor/weights", agent.actor, t)
        # log_grad_hist(f"model/actor/grads", agent.actor, t)

        # log_weight_hist(f"model/q1/weights", agent.Q1, t)
        # log_grad_hist(f"model/q1/grads", agent.Q1, t)

        # log_weight_hist(f"model/q2/weights", agent.Q2, t)
        # log_grad_hist(f"model/q2/grads", agent.Q2, t)

        # Periodic eval
        if t % eval_interval == 0:
            save_ckpt(agent, f"sac_osim_step{t}.pt")

            agent.eval()
            avg_ret = evaluate(model, pose, agent, t, episodes=5)
            print(f"[{t:6d}] eval_return={avg_ret:.2f}  alpha={agent.get_alpha():.4f}")
            agent.train()

    # Final eval
    final_ret = evaluate(model, pose, agent, -1, episodes=10)
    print(f"Final average return over 10 episodes: {final_ret:.2f}")

    # Save checkpoint
    save_ckpt(agent, "sac_osim.pt")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        mlflow.end_run()  # mlflow.end_run is idempotent
        clear_tmp()

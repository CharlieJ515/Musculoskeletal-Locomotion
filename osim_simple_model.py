from typing import Tuple, cast
from pathlib import Path

import mlflow
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gymnasium as gym
import opensim

from environment.models import gait14dof22_path
from environment.osim import OsimEnv
from environment.osim.pose import Pose, osim_rl_pose
from environment.osim.reward import (
    CompositeReward,
    AliveReward,
    VelocityReward,
    EnergyReward,
    SmoothnessReward,
)
from environment.wrappers import (
    TargetSpeedWrapper,
    MotionLoggerWrapper,
    SimpleEnvWrapper,
    CompositeRewardWrapper,
)
from rl.replay_buffer.replay_buffer import ReplayBuffer
from rl.sac import SAC, default_target_entropy
from utils.transition import Transition, TransitionBatch
from utils.mlflow_utils import get_tmp, tag_attempt, clear_tmp
from analysis.writer import get_writer
from analysis.distribution import log_weight_hist, log_grad_hist, log_rewards


class MLPActor(nn.Module):
    def __init__(
        self,
        state_dim: Tuple[int, ...],
        action_dim: Tuple[int, ...],
        log_std_min: float,
        log_std_max: float,
    ):
        super().__init__()
        self.state_dim = int(np.prod(state_dim))
        self.action_dim = int(np.prod(action_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        hidden_dims = [1024, 1024, 512, 256]

        layers = []
        input_dim = self.state_dim
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            input_dim = h

        self.net = nn.Sequential(*layers)
        self.mu_head = nn.Linear(hidden_dims[-1], self.action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], self.action_dim)

    def forward(self, s: torch.Tensor, deterministic: bool = False):
        x = s.view(s.shape[0], -1)
        h = self.net(x)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()

        if deterministic:
            z = mu
            log_prob = torch.zeros(s.size(0), 1, device=s.device)
        else:
            dist = torch.distributions.Normal(mu, std)
            z = dist.rsample()  # reparameterization
            # log prob of pre-tanh sample
            log_prob = dist.log_prob(z).sum(dim=-1, keepdim=True)

        # Tanh squashing with correction (when stochastic)
        a = torch.tanh(z)
        if not deterministic:
            # change-of-variables correction: log(1 - tanh(z)^2)
            log_prob -= torch.log(1 - a.pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        return a, log_prob


class MLPCritic(nn.Module):
    def __init__(
        self,
        state_dim: Tuple[int, ...],
        action_dim: Tuple[int, ...],
        reward_dim: Tuple[int, ...],
    ):
        super().__init__()
        self.state_dim = int(np.prod(state_dim))
        self.action_dim = int(np.prod(action_dim))
        self.reward_dim = int(np.prod(reward_dim))

        hidden_dims = [1024, 1024, 512, 256]

        layers = []
        input_dim = self.state_dim + self.action_dim
        for h in hidden_dims:
            layers.append(nn.Linear(input_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            input_dim = h
        layers.append(nn.Linear(input_dim, self.reward_dim))

        self.q = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        x = torch.cat([s.view(s.shape[0], -1), a.view(a.shape[0], -1)], dim=-1)
        return self.q(x)


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

    agent.train(True)
    return float(np.mean(returns))


def save_ckpt(agent: SAC, ckpt_name: str):
    tmpdir = get_tmp()
    if not ckpt_name.endswith(".pt"):
        ckpt_name = f"{ckpt_name}.pt"
    ckpt_path = tmpdir / ckpt_name
    agent.save(ckpt_path)
    mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")
    print(f"Saved checkpoint")


def random_action_gamma_dist(
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


def pretrain_actor(
    rb: ReplayBuffer,
    actor,
    name: str = "actor",
    num_epoch: int = 10,
    batch_size: int = 256,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    lr: float = 0.001,
):
    actor.to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=lr)
    num_step = max(1, len(rb) // batch_size)

    print("Starting actor pretraining")
    for epoch in range(1, num_epoch + 1):
        loss_sum = 0.0
        for step in range(1, num_step + 1):
            batch = rb.sample(256, pin_memory=True).to(device)
            s, a, _, _, _ = batch.unpack()

            action, _ = actor(s, deterministic=True)
            loss = F.mse_loss(action, a)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.detach().cpu().item()

            t = num_step * epoch + step
            # writer.add_histogram(f"pretrain/{name}/output", action.detach().cpu(), t)
            # writer.add_histogram(f"pretrain/{name}/replay_buffer", a, t)

            # log_weight_hist(f"pretrain/actor/weights", actor, t)
            # log_grad_hist(f"pretrain/actor/grads", actor, t)

        epoch_loss = loss_sum / num_step
        print(f"epoch {epoch:3}, loss: {epoch_loss:.4f}")

    # s, a, _, _, _ = rb.all(pin_memory=True).to(device).unpack()
    # action, _ = actor(s, deterministic=True)

    # writer.add_histogram(f"pretrain/{name}/total/replay_buffer", a)
    # writer.add_histogram(f"pretrain/{name}/total/output", action.detach().cpu())

    print("Actor pretraining complete")


def pretrain_critic(
    rb: ReplayBuffer,
    critic,
    name: str = "critic",
    num_epoch: int = 10,
    batch_size: int = 256,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    lr: float = 1e-3,
):
    critic.to(device)
    optimizer = torch.optim.Adam(critic.parameters(), lr=lr)
    num_step = max(1, len(rb) // batch_size)

    print("Starting critic pretraining")
    for epoch in range(1, num_epoch + 1):
        loss_sum = 0.0
        for step in range(1, num_step + 1):
            batch = rb.sample(batch_size, pin_memory=True).to(device)
            s, a, r, _, _ = batch.unpack()

            q_pred = critic(s, a)
            loss = F.mse_loss(q_pred, r)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_sum += loss.detach().cpu().item()

            t = num_step * epoch + step
            # writer.add_histogram(f"pretrain/{name}/output", q_pred.detach().cpu(), t)
            # writer.add_histogram(f"pretrain/{name}/replay_buffer", r, t)

            # log_weight_hist(f"pretrain/{name}/weights", critic, t)
            # log_grad_hist(f"pretrain/{name}/grads", critic, t)

        epoch_loss = loss_sum / num_step
        print(f"epoch {epoch:3}, loss: {epoch_loss:.4f}")

    # s, a, r, _, _ = rb.all(pin_memory=True).to(device).unpack()
    # q_pred = critic(s, a).detach().cpu()

    # plt.figure(figsize=(8, 6))
    # colors = ["r", "g", "b", "orange"]  # one color per distribution
    # labels = [f"Q{i}" for i in range(4)]

    # for i in range(4):
    #     plt.hist(q_pred[:, i], bins=50, alpha=0.5, color=colors[i], label=labels[i])

    # plt.title("Q-value Distributions")
    # plt.xlabel("Q value")
    # plt.ylabel("Frequency")
    # plt.legend()
    # plt.grid(True, linestyle="--", alpha=0.7)
    # plt.show()

    # writer.add_histogram(f"pretrain/{name}/replay_buffer", a)
    # writer.add_histogram(f"pretrain/{name}/output", q_pred)

    print("Critic pretraining complete")


def create_env(model: Path, pose: Pose) -> gym.Env:
    osim_env = OsimEnv(model, pose, visualize=True)
    time_limit_env = gym.wrappers.TimeLimit(osim_env, 500)
    target_env = TargetSpeedWrapper(time_limit_env)
    reward_components = {
        "alive_reward": AliveReward(0.1),
        "velocity_reward": VelocityReward(0.33),
        "energy_reward": EnergyReward(3.0),
        "smoothness_reward": SmoothnessReward(2.0),
    }
    reward_weights = {
        "alive_reward": 1.0,
        "velocity_reward": 1.0,
        "energy_reward": 1.0,
        "smoothness_reward": 1.0,
    }
    reward_fn = CompositeReward(reward_components, reward_weights)
    reward_env = CompositeRewardWrapper(target_env, reward_fn)
    simple_env = SimpleEnvWrapper(reward_env)
    time_aware_env = gym.wrappers.TimeAwareObservation(
        simple_env, flatten=True, normalize_time=True
    )
    rescale_env = gym.wrappers.RescaleAction(
        time_aware_env, np.float32(-1.0), np.float32(1.0)
    )
    return rescale_env


def reward_info_to_ndarray(reward_info: dict[str, np.ndarray]) -> np.ndarray:
    reward = np.array(
        [
            reward_info["alive_reward"],
            reward_info["velocity_reward"],
            reward_info["energy_reward"],
            reward_info["smoothness_reward"],
        ],
        dtype=np.float32,
    ).T
    return reward


def main():
    seed = 42
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_name = "SAC-Osim3"
    run_name = "simple sac"
    model = gait14dof22_path
    pose = osim_rl_pose

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

    env = gym.vector.AsyncVectorEnv(
        [lambda: create_env(model, pose) for _ in range(2)],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
    )

    obs_shape = cast(Tuple[int, ...], env.single_observation_space.shape)
    action_shape = cast(Tuple[int, ...], env.single_action_space.shape)
    reward_shape = (4,)
    reward_weights = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
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
        target_entropy=-22,  # std ~0.25
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
        capacity=5_000,
        obs_shape=obs_shape,
        action_shape=action_shape,
        reward_shape=reward_shape,
        device=torch.device("cpu"),
    )
    rb.log_params(prefix="buffer/")

    # Training loop
    total_steps = 25_000
    # total_steps = 1_000
    start_random = 1_000
    batch_size = 256
    eval_interval = 5_000
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
        a_np = random_action_gamma_dist(
            env.action_space  # pyright: ignore[reportArgumentType]
        )
        s_next_np, r, terminated, truncated, info = env.step(a_np)

        if episode_start.all():
            s_np = s_next_np
            episode_start = np.logical_or(terminated, truncated)
            continue

        reward_info = info["rewards"]
        reward = reward_info_to_ndarray(reward_info)

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
    actor = agent.actor
    q1 = agent.Q1
    q2 = agent.Q2
    pretrain_actor(rb, actor, device=device)
    pretrain_critic(rb, q1, device=device, name="q1")
    pretrain_critic(rb, q2, device=device, name="q2")
    agent._hard_update()

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
        reward = reward_info_to_ndarray(reward_info)

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
            writer.add_histogram("train/critic/q1_pred/alive", q1_pred[:, 0], t)
            writer.add_histogram("train/critic/q1_pred/velocity", q1_pred[:, 1], t)
            writer.add_histogram("train/critic/q1_pred/energy", q1_pred[:, 2], t)
            writer.add_histogram("train/critic/q1_pred/smoothness", q1_pred[:, 3], t)

            q2_pred = critic_metrics["q2_pred"]
            writer.add_histogram("train/critic/q2_pred/alive", q2_pred[:, 0], t)
            writer.add_histogram("train/critic/q2_pred/velocity", q2_pred[:, 1], t)
            writer.add_histogram("train/critic/q2_pred/energy", q2_pred[:, 2], t)
            writer.add_histogram("train/critic/q2_pred/smoothness", q2_pred[:, 3], t)

            q_target = critic_metrics["q_target"]
            writer.add_histogram("train/critic/q_target/alive", q_target[:, 0], t)
            writer.add_histogram("train/critic/q_target/velocity", q_target[:, 1], t)
            writer.add_histogram("train/critic/q_target/energy", q_target[:, 2], t)
            writer.add_histogram("train/critic/q_target/smoothness", q_target[:, 3], t)

            writer.add_histogram("train/actor/log_prob", actor_metrics["log_prob"], t)
            writer.add_histogram("train/actor/q", actor_metrics["q"], t)

        # writer.add_scalars("transit/reward", infos[0]["rewards"], t)

        # ---

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


if __name__ == "__main__":
    try:
        main()
    finally:
        mlflow.end_run()  # mlflow.end_run is idempotent
        clear_tmp()

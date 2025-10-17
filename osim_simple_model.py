from pathlib import Path
from typing import Optional, Tuple, cast  

import mlflow
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box
import opensim

from environment import L2M2019GymEnv
from environment.wrappers import ScaleAction
from environment.models import gait14dof22musc
from rl.replay_buffer.replay_buffer import ReplayBuffer
from rl.sac import SAC, default_target_entropy
from utils.transition import Transition
from utils.mlflow_utils import get_tmp, tag_attempt, clear_tmp



class MLPActor(nn.Module):
    def __init__(self, state_dim: Tuple[int, ...], action_dim: Tuple[int, ...], log_std_min: float, log_std_max: float):
        super().__init__()
        self.state_dim = int(np.prod(state_dim))
        self.action_dim = int(np.prod(action_dim))
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        hidden = 256
        self.net = nn.Sequential(
            nn.Linear(self.state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden, self.action_dim)
        self.log_std_head = nn.Linear(hidden, self.action_dim)

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
    def __init__(self, state_dim: Tuple[int, ...], action_dim: Tuple[int, ...]):
        super().__init__()
        self.state_dim = int(np.prod(state_dim))
        self.action_dim = int(np.prod(action_dim))
        hidden = 256
        self.q = nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        x = torch.cat([s.view(s.shape[0], -1), a.view(a.shape[0], -1)], dim=-1)
        return self.q(x)  # (B,1)


@torch.no_grad()
def evaluate(
    env: gym.Env,
    agent: SAC,
    step: int,
    episodes: int = 5,
):
    active_run = mlflow.active_run()
    active_run_name = active_run.data.tags.get("mlflow.runName") # type: ignore[reportOptionalMemberAccess]

    agent.eval()
    gamma = agent.gamma

    tmpdir = get_tmp()
    returns = []

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

        s, info = env.reset()
        done = False
        ep_step = 0
        while not done:
            st = torch.tensor(s, dtype=torch.float32, device=agent.device).unsqueeze(0)

            a = agent.select_action(st)
            a_np = a.squeeze(0).cpu().numpy()
            s_next, r, terminated, truncated, info = env.step(a_np)
            done = terminated or truncated
            ep_ret += float(r)

            ep_disc += disc_pow * float(r)
            disc_pow *= float(gamma)

            q1 = agent.Q1(st, a).detach().cpu().item()
            q2 = agent.Q2(st, a).detach().cpu().item()
            qmin = min(q1, q2)

            mlflow.log_metrics(
                {
                    "reward": float(r),
                    "cumulative_return": float(ep_ret),
                    "discounted_return": float(ep_disc),
                    "Q1": float(q1),
                    "Q2": float(q2),
                    "Qmin": float(qmin),
                },
                step=ep_step
            )

            all_actions.append(a_np)

            s = s_next
            ep_step += 1

        # log total returns
        mlflow.log_metrics(
            {
                "total_return": float(ep_ret),
                "total_discounted_return": float(ep_disc),
                "episode_length": int(ep_step),
            },
            step=ep_step
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

def save_ckpt(agent:SAC, ckpt_name:str):
    tmpdir = get_tmp()
    ckpt_path = tmpdir / f"{ckpt_name}.pt"
    agent.save(ckpt_path)
    mlflow.log_artifact(str(ckpt_path), artifact_path="checkpoints")
    print(f"Saved checkpoint")

def main():
    seed = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    experiment_name = "SAC-Osim"
    run_name = "simple sac"
    model = gait14dof22musc
    difficulty = 0

    mlflow.set_tracking_uri("https://mlflow.kyusang-jang.com/capstone")
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name)
    tag_attempt(experiment_name, run_name)

    mlflow.log_params({
        "env_id": "L2M2019Env",
        "opensim_model": str(model),
        "difficulty": difficulty,
        "seed": seed,
        "gym_version": gym.__version__,
        "opensim_version": opensim.__version__,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    })

    torch.manual_seed(seed)
    np.random.seed(seed)
    env = L2M2019GymEnv(model_path=model, difficulty=difficulty, visualize=False, seed=seed)

    # wrap env with ScaleAction action wrapper
    act_space = cast(Box, env.action_space)
    act_type = act_space.dtype.type
    src_low = -np.ones_like(act_space.low,  dtype=act_type)
    src_high = np.ones_like(act_space.high, dtype=act_type)
    source_space = Box(low=src_low, high=src_high, dtype=act_type)
    env = ScaleAction(env, source_space)
    env.action_space.seed(seed)

    # log observation / action space size and range
    # log after wrapping because wrapper modifies the action_space
    obs_shape = cast(Tuple[int, ...], env.observation_space.shape)
    act_shape = cast(Tuple[int, ...], env.action_space.shape)
    rew_shape = (1, )

    mlflow.log_params({
        "obs_shape": obs_shape,
        "act_shape": act_shape,
        "rew_shape": rew_shape,
    })

    # SAC hyperparams
    agent = SAC(
        gamma=0.99,
        state_dim=obs_shape,
        action_dim=act_shape,
        reward_dim=rew_shape,
        log_std_min=-20.0,
        log_std_max=2.0,
        actor_net=MLPActor,
        critic_net=MLPCritic,
        lr=3e-4,
        tau=0.005,
        target_entropy=default_target_entropy(act_shape),
        weight_decay=0.0,
        policy_update_freq=1,
        reward_weight=torch.ones(1),
        device=device,
        use_jit=False,
        train=True,
    )
    agent.log_params(prefix="agent/")

    # Replay buffer
    rb = ReplayBuffer(
        capacity=200_000,
        obs_shape=obs_shape,
        action_shape=act_shape,
        reward_shape=rew_shape,
        device=torch.device("cpu"),
    )
    rb.log_params(prefix="buffer/")

    # Training loop
    total_steps = 25_000
    start_random = 1_000
    batch_size = 256
    eval_interval = 5_000
    mlflow.log_params({
        "total_steps": total_steps,
        "start_random": start_random,
        "batch_size": batch_size,
        "eval_interval": eval_interval,
    })

    agent.train()
    s, info = env.reset(seed=seed)
    ep_return, ep_len = 0.0, 0
    for t in range(1, total_steps + 1):
        if t < start_random:
            a = env.action_space.sample()
        else:
            st = torch.tensor(s, dtype=torch.float32, device=agent.device).unsqueeze(0)
            a_t = agent.select_action(st)
            a = a_t.squeeze(0).cpu().numpy()

        s_next, r, terminated, truncated, info = env.step(a)
        done = terminated or truncated
        ep_return += float(r); ep_len += 1

        # Store transition
        tr = Transition(
            obs=torch.as_tensor(s, dtype=torch.float32),
            action=torch.as_tensor(a, dtype=torch.float32),
            reward=torch.as_tensor([r], dtype=torch.float32),
            next_obs=torch.as_tensor(s_next, dtype=torch.float32),
            done=torch.as_tensor(done, dtype=torch.bool),
        )
        rb.add(tr)

        s = s_next

        # Update
        if t >= start_random:
            batch = rb.sample(batch_size, pin_memory=True).to(device, non_blocking=True)
            metrics = agent.update(batch)
            mlflow.log_metrics(metrics, step=t)

        # Episode reset
        if done:
            print(f"Episode finished at step {t:6d} | length={ep_len} | return={ep_return:.2f}")
            s, info = env.reset()
            ep_return, ep_len = 0.0, 0

        # Periodic eval
        if t % eval_interval == 0:
            save_ckpt(agent, f"sac_osim_step{t}.pt")

            agent.eval()
            avg_ret = evaluate(env, agent, t, episodes=5)
            print(f"[{t:6d}] eval_return={avg_ret:.2f}  alpha={agent.get_alpha():.4f}")
            agent.train()

    # Final eval
    final_ret = evaluate(env, agent, -1, episodes=10)
    print(f"Final average return over 10 episodes: {final_ret:.2f}")

    # Save checkpoint
    save_ckpt(agent, "sac_osim.pt")

if __name__ == "__main__":
    try:
        main()
    finally:
        mlflow.end_run() # mlflow.end_run is idempotent
        clear_tmp()

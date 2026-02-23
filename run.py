from typing import cast

import gymnasium as gym
import matplotlib
import numpy as np
import torch
import yaml
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis import MlflowWriter, TBWriter
from configs import PERConfig, TD3Config, TrainConfig
from environment.builder import create_env
from environment.wrappers import MotionLoggerWrapper, reward_info_to_ndarray
from models.td3_mlp import MLPActor, MLPCritic
from rl import PER, TD3, BaseRL, TransitionBatch, build_noise
from utils.save import save_ckpt
from utils.tmp_dir import clear_tmp, get_tmp


@torch.no_grad()
def evaluate(
    cfg: TrainConfig,
    agent: BaseRL,
    step: int,
    episodes: int,
    mlflow_writer: MlflowWriter,
):
    active_run_name = mlflow_writer.active_run_name()

    tmpdir = get_tmp()
    returns = []

    motion_output_dir = tmpdir / f"{active_run_name}_eval_step{step}"
    env = create_env(cfg, False)
    env = MotionLoggerWrapper(
        env,
        motion_output_dir,
        f"{active_run_name}_eval_step{step}_{{}}.mot",
    )
    agent.eval()

    device = getattr(agent, "device", torch.device("cpu"))
    gamma = getattr(agent, "gamma", 0.99)

    for ep in range(episodes):
        run_name = f"{active_run_name}_eval_step{step}_{ep}"
        eval_dir = tmpdir / run_name
        eval_dir.mkdir(parents=True, exist_ok=True)

        with mlflow_writer.nested_run(run_name):
            mlflow_writer.set_tags(
                {"phase": "eval", "checkpoint_step": step, "episode": ep}
            )

            all_actions = []
            ep_ret = 0.0
            ep_disc = 0.0
            disc_pow = 1.0

            s_np, info = env.reset()
            ep_step = 0
            while True:
                s_t = torch.tensor(s_np, dtype=torch.float32, device=device).unsqueeze(
                    0
                )

                a_t = agent.select_action(s_t)
                a_np = a_t.squeeze(0).cpu().numpy()
                s_next_np, r, terminated, truncated, info = env.step(a_np)
                ep_ret += float(r)

                ep_disc += disc_pow * float(r)
                disc_pow *= float(gamma)

                metrics = {
                    "cumulative_return": float(ep_ret),
                    "discounted_return": float(ep_disc),
                    **info["rewards"],
                }

                if hasattr(agent, "Q1") and hasattr(agent, "Q2"):
                    q1 = agent.Q1(s_t, a_t).detach().cpu().squeeze(0)  # type: ignore
                    q2 = agent.Q2(s_t, a_t).detach().cpu().squeeze(0)  # type: ignore
                    qmin = torch.min(q1, q2)
                    for i, key in enumerate(cfg.reward_key):
                        metrics[f"q1_pred/{key}"] = q1[i]
                        metrics[f"q2_pred/{key}"] = q2[i]
                        metrics[f"min/{key}"] = qmin[i]

                mlflow_writer.log_metrics(metrics, step=ep_step)
                all_actions.append(a_np)

                s_np = s_next_np
                ep_step += 1

                if terminated or truncated:
                    mot_path = info["mot_path"]
                    mlflow_writer.log_artifact(str(mot_path), artifact_path="motion")
                    break

            mlflow_writer.log_metrics(
                {
                    "total_return": float(ep_ret),
                    "total_discounted_return": float(ep_disc),
                    "episode_length": int(ep_step),
                },
                step=ep_step,
            )

            A = np.asarray(all_actions)
            A_flat = A.reshape(-1)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist(A_flat, bins=100, log=True)
            ax.set_title(f"Action distribution step {step}, eval ep {ep}")
            ax.set_xlabel("action value")
            ax.set_ylabel("count (log)")
            mlflow_writer.log_figure(fig, f"figures/action_hist_step{step}_ep{ep}.png")
            plt.close(fig)

            returns.append(ep_ret)

    env.close()
    agent.train(True)
    return float(np.mean(returns))


def main(
    cfg: TrainConfig,
    agent: BaseRL,
    rb: PER,
    tb_writer: TBWriter,
    mlflow_writer: MlflowWriter,
):
    env = gym.vector.AsyncVectorEnv(
        [lambda: create_env(cfg, False) for _ in range(cfg.num_env)],
        autoreset_mode=gym.vector.AutoresetMode.NEXT_STEP,
        context=cfg.mp_context,
    )
    device = agent.device

    agent.log_params(mlflow_writer, prefix="agent/")
    rb.log_params(mlflow_writer, prefix="buffer/")

    act_shape = cast(tuple[int, ...], env.single_action_space.shape)
    noise_sampler = build_noise(cfg.noise, cfg.num_env, act_shape)

    s_np, infos = env.reset(seed=cfg.seed)
    episode_start = np.array([False] * env.num_envs, np.bool_)

    tb_writer.set_logging(histograms=False)

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
    episode_start = np.array([False] * env.num_envs, np.bool_)

    print("Starting training")
    for t in tqdm(range(1, cfg.total_steps + 1)):
        s_t = torch.as_tensor(s_np, dtype=torch.float32, device=device)
        a_t = agent.select_action(s_t)
        a_np = a_t.cpu().numpy()
        noise = noise_sampler.sample()
        a_np = (a_np + noise).clip(-1.0, 1.0)

        env.step_async(a_np)

        batch = rb.sample(cfg.batch_size, pin_memory=True).to(device, non_blocking=True)
        metrics = agent.update(batch)

        td_error = metrics.get("critic", {}).get("td_error", metrics.get("td_error"))
        if td_error is not None:
            rb.update_priorities(batch.indices, td_error)  # type: ignore

        s_next_np, r, terminated, truncated, info = env.step_wait()

        if episode_start.all():
            s_np = s_next_np
            episode_start = np.logical_or(terminated, truncated)
            noise_sampler.reset()
            continue

        reward_info = info["rewards"]
        reward = reward_info_to_ndarray(cfg.reward_key, reward_info)

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
        should_log_hist = t % cfg.log_interval == 0
        tb_writer.set_logging(histograms=should_log_hist)

        agent.write_logs(metrics, t, tb_writer, mlflow_writer)
        tb_writer.log_rewards(reward_info, idx, t, "transit/reward")

        # eval
        if t % cfg.eval_interval == 0:
            save_ckpt(agent, f"agent_osim_step{t}.pt")

            agent.eval()
            avg_ret = evaluate(cfg, agent, t, cfg.eval_episodes, mlflow_writer)
            print(f"[{t:6d}] eval_return={avg_ret:.2f}")
            agent.train()

    # Final eval
    final_ret = evaluate(cfg, agent, -1, cfg.eval_episodes, mlflow_writer)
    print(f"Final average return over {cfg.eval_episodes} episodes: {final_ret:.2f}")

    save_ckpt(agent, "agent_osim.pt")
    env.close()


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        yaml_config = yaml.safe_load(f)

    cfg = TrainConfig.from_dict(yaml_config["train"])

    agent_cfg = TD3Config.from_dict(
        yaml_config["td3"], actor_net=MLPActor, critic_net=MLPCritic
    )
    agent = TD3.from_config(agent_cfg)
    per_cfg = PERConfig.from_dict(yaml_config["per"])
    rb = PER.from_config(per_cfg)

    # mlflow
    mlflow_writer = MlflowWriter()
    mlflow_writer.start_main_run(
        uri="https://mlflow.kyusang-jang.com/capstone",
        experiment_name="TD3-Osim",
        run_name="td3_body-basic-reward2",
    )

    # tensorboard
    active_run = mlflow_writer.active_run_name()
    tb_writer = TBWriter(log_dir=f"runs/{active_run}")

    try:
        main(cfg, agent, rb, tb_writer, mlflow_writer)
    finally:
        mlflow_writer.end_main_run()
        clear_tmp()

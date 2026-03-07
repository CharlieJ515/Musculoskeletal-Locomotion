import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

from rl.replay_buffer.prioritized_replay_buffer import (
    PrioritizedReplayBuffer,
    PrioritizedReplayConfig,
)
from rl.transition import Transition
from rl.td3 import TD3


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.l1 = nn.Linear(state_dim[0], 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim[0])
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, reward_dim):
        super().__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim[0] + action_dim[0], 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        q = torch.cat([state, action], 1)
        q = torch.relu(self.l1(q))
        q = torch.relu(self.l2(q))
        return self.l3(q)


def main():
    env_name = "Pendulum-v1"
    env = gym.make(env_name, render_mode="human")

    state_dim = (env.observation_space.shape[0],)
    action_dim = (env.action_space.shape[0],)
    max_action = float(env.action_space.high[0])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Training on {env_name} | Device: {device}")

    # 2. Initialize Agent
    agent = TD3(
        gamma=0.99,
        state_dim=state_dim,
        action_dim=action_dim,
        reward_dim=(1,),
        actor_net=Actor,
        critic_net=Critic,
        lr=3e-4,
        tau=0.005,
        weight_decay=0.0,
        policy_noise=0.2,
        noise_clip=0.5,
        max_action=max_action,
        exploration_noise=0.1,
        device=device,
        use_jit=True,
    )

    buffer_cfg = PrioritizedReplayConfig(
        capacity=50_000,
        obs_shape=state_dim,
        action_shape=action_dim,
        reward_shape=(1,),
        alpha=0.6,
        beta_start=0.4,
        beta_frames=10_000,
        device=device,
        obs_dtype=torch.float32,
        action_dtype=torch.float32,
        reward_dtype=torch.float32,
    )
    buffer = PrioritizedReplayBuffer.from_config(buffer_cfg)

    # 4. Training Loop
    episodes = 200
    batch_size = 256
    start_steps = 1000
    total_steps = 0

    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        truncated = False

        while not (done or truncated):
            total_steps += 1

            # A. Select Action
            if total_steps < start_steps:
                action = env.action_space.sample()
            else:
                s_tensor = torch.tensor(
                    state, dtype=torch.float32, device=device
                ).unsqueeze(0)
                action = agent.select_action(s_tensor).cpu().numpy().flatten()

                noise = np.random.normal(0, 0.1, size=action_dim)
                action = (action + noise).clip(-max_action, max_action)

            # B. Step Environment
            next_state, reward, terminated, truncated, _ = env.step(action)

            # C. Add to Buffer
            t = Transition(
                obs=torch.tensor(state, dtype=torch.float32),
                action=torch.tensor(action, dtype=torch.float32),
                reward=torch.tensor([reward], dtype=torch.float32),
                next_obs=torch.tensor(next_state, dtype=torch.float32),
                done=torch.tensor(terminated, dtype=torch.bool),
            )
            buffer.add(t)

            state = next_state
            episode_reward += float(reward)

            # D. Update Agent
            if total_steps >= start_steps:
                # 1. Sample (returns TransitionBatch with .weights and .indices)
                batch = buffer.sample(batch_size)

                # 2. Update Networks
                metrics = agent.update(batch)

                # 3. Update Priorities
                td_error = metrics["critic"]["td_error"]
                buffer.update_priorities(batch.indices, td_error)

        print(
            f"Episode {episode+1}: Reward = {episode_reward:.2f} | Total Steps: {total_steps}"
        )


if __name__ == "__main__":
    main()

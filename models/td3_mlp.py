import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLPActor(nn.Module):
    def __init__(
        self,
        state_dim: tuple[int, ...],
        action_dim: tuple[int, ...],
    ):
        super().__init__()
        self.state_dim = int(np.prod(state_dim))
        self.action_dim = int(np.prod(action_dim))

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

        self._init_layers()

    def _init_layers(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        nn.init.constant_(self.mu_head.bias, 0.0)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = s.view(s.shape[0], -1)
        h = self.net(x)
        mu = self.mu_head(h)
        action = torch.tanh(mu)
        return action


class MLPCritic(nn.Module):
    def __init__(
        self,
        state_dim: tuple[int, ...],
        action_dim: tuple[int, ...],
        reward_dim: tuple[int, ...],
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

        self.net = nn.Sequential(*layers)
        self.q_head = nn.Linear(input_dim, self.reward_dim)

        self._init_layers()

    def _init_layers(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)

        nn.init.orthogonal_(self.q_head.weight, gain=1.0)
        nn.init.constant_(self.q_head.bias, 0.0)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s.view(s.shape[0], -1), a.view(a.shape[0], -1)], dim=-1)
        x = self.net(x)
        q_value = self.q_head(x)
        return q_value

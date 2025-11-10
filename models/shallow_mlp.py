import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLPActor(nn.Module):
    def __init__(
        self,
        state_dim: tuple[int, ...],
        action_dim: tuple[int, ...],
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
        layers.append(nn.Linear(input_dim, self.reward_dim))

        self.q = nn.Sequential(*layers)

    def forward(self, s: torch.Tensor, a: torch.Tensor):
        x = torch.cat([s.view(s.shape[0], -1), a.view(a.shape[0], -1)], dim=-1)
        return self.q(x)

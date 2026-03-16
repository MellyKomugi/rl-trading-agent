from __future__ import annotations

import numpy as np
import torch
from torch import nn


class SimplePortfolioMLP(nn.Module):
    """Simple portfolio policy that maps state + previous allocation to weights."""

    def __init__(
        self,
        input_shape=(7, 3),
        portfolio_size=3,
        hidden_dim=64,
        dropout=0.1,
        device="cpu",
    ):
        super().__init__()
        self.input_shape = input_shape
        self.portfolio_size = portfolio_size
        self.device = device

        input_dim = (input_shape[0] * input_shape[1]) + portfolio_size
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, portfolio_size)

    def _as_tensor(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(self.device, dtype=torch.float32)
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _prepare_last_action(self, last_action, batch_size):
        if last_action is None:
            return torch.zeros(
                batch_size, self.portfolio_size, dtype=torch.float32, device=self.device
            )

        last_action = self._as_tensor(last_action)
        if last_action.dim() == 1:
            last_action = last_action.unsqueeze(0)

        # FinRL portfolio policies sometimes include a cash slot. Drop it here.
        if last_action.shape[-1] == self.portfolio_size + 1:
            last_action = last_action[:, 1:]

        return last_action

    def mu(self, obs, last_action=None):
        obs = self._as_tensor(obs)
        if obs.dim() == 2:
            obs = obs.unsqueeze(0)

        batch_size = obs.shape[0]
        obs = obs.reshape(batch_size, -1)
        last_action = self._prepare_last_action(last_action, batch_size)

        features = torch.cat([obs, last_action], dim=1)
        logits = self.head(self.encoder(features))
        return torch.softmax(logits, dim=1)

    def forward(self, obs, last_action=None):
        allocation = self.mu(obs, last_action)
        if isinstance(obs, torch.Tensor):
            return allocation
        return allocation.squeeze(0).detach().cpu().numpy()


# Compatibility alias for the current PolicyGradient scaffold.
EIIE = SimplePortfolioMLP

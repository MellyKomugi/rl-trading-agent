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

class EIIE(nn.Module):
    def __init__(
        self,
        initial_features=3,
        k_size=3,
        conv_mid_features=2,
        conv_final_features=20,
        time_window=50,
        device="cpu",
    ):
        """EIIE (ensemble of identical independent evaluators) policy network
        initializer.

        Args:
            initial_features: Number of input features.
            k_size: Size of first convolutional kernel.
            conv_mid_features: Size of intermediate convolutional channels.
            conv_final_features: Size of final convolutional channels.
            time_window: Size of time window used as agent's state.
            device: Device in which the neural network will be run.

        Note:
            Reference article: https://doi.org/10.48550/arXiv.1706.10059.
        """
        super().__init__()
        self.device = device

        n_size = time_window - k_size + 1

        self.sequential = nn.Sequential(
            nn.Conv2d(
                in_channels=initial_features,
                out_channels=conv_mid_features,
                kernel_size=(1, k_size),
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=conv_mid_features,
                out_channels=conv_final_features,
                kernel_size=(1, n_size),
            ),
            nn.ReLU(),
        )

        self.final_convolution = nn.Conv2d(
            in_channels=conv_final_features + 1, out_channels=1, kernel_size=(1, 1)
        )

        self.softmax = nn.Sequential(nn.Softmax(dim=-1))

    def mu(self, observation, last_action):
        """Defines a most favorable action of this policy given input x.

        Args:
          observation: environment observation.
          last_action: Last action performed by agent.

        Returns:
          Most favorable action.
        """

        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation)
        observation = observation.to(self.device).float()

        if isinstance(last_action, np.ndarray):
            last_action = torch.from_numpy(last_action)
        last_action = last_action.to(self.device).float()

        last_stocks, cash_bias = self._process_last_action(last_action)
        cash_bias = torch.zeros_like(cash_bias).to(self.device)

        output = self.sequential(observation)  # shape [N, 20, PORTFOLIO_SIZE, 1]
        output = torch.cat(
            [last_stocks, output], dim=1
        )  # shape [N, 21, PORTFOLIO_SIZE, 1]
        output = self.final_convolution(output)  # shape [N, 1, PORTFOLIO_SIZE, 1]
        output = torch.cat(
            [cash_bias, output], dim=2
        )  # shape [N, 1, PORTFOLIO_SIZE + 1, 1]

        # output shape must be [N, features] = [1, PORTFOLIO_SIZE + 1], being N batch size (1)
        # and size the number of features (weights vector).
        output = torch.squeeze(output, 3)
        output = torch.squeeze(output, 1)  # shape [N, PORTFOLIO_SIZE + 1]

        output = self.softmax(output)

        return output

    def forward(self, observation, last_action):
        """Policy network's forward propagation.

        Args:
          observation: Environment observation (dictionary).
          last_action: Last action performed by the agent.

        Returns:
          Action to be taken (numpy array).
        """
        mu = self.mu(observation, last_action)
        action = mu.cpu().detach().numpy().squeeze()
        return action

    def _process_last_action(self, last_action):
        """Process the last action to retrieve cash bias and last stocks.

        Args:
          last_action: Last performed action.

        Returns:
            Last stocks and cash bias.
        """
        batch_size = last_action.shape[0]
        stocks = last_action.shape[1] - 1
        last_stocks = last_action[:, 1:].reshape((batch_size, 1, stocks, 1))
        cash_bias = last_action[:, 0].reshape((batch_size, 1, 1, 1))
        return last_stocks, cash_bias


# Compatibility alias for the current PolicyGradient scaffold.
# EIIE = SimplePortfolioMLP

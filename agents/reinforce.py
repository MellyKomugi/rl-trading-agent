# agents/reinforce.py
#
# REINFORCE (Monte Carlo Policy Gradient) agent for portfolio allocation.
#
# Design choices:
#   - Policy network: MLP with two hidden layers.
#       Input  : flattened observation matrix (7 × 3 = 21 features).
#       Output : 3 raw logits → softmax → portfolio weights [w_AAPL, w_JPM, w_XOM].
#   - Stochastic policy via Dirichlet distribution.
#       Dirichlet is the natural choice for portfolio weights: it samples
#       vectors that are strictly positive and sum to 1, which matches the
#       action space of StockPortfolioEnv exactly. The network outputs
#       concentration parameters (α > 0) via softplus.
#   - Returns: Monte Carlo (full episode) with reward-to-go and baseline.
#       Subtracting the mean return (baseline) reduces variance without
#       introducing bias — standard practice for REINFORCE.
#   - Reward scaling: applied here (× 1e-4), not inside the env, as
#       instructed by setup_env.py comments and config.py.
#   - The env uses the old gym API: step() returns (obs, reward, done, info).
#       No truncated flag.

from environment.setup_env import build_envs

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Dirichlet

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import REWARD_SCALING, RANDOM_SEED


# ── Gym API compatibility helpers ─────────────────────────────────────────────
# StockPortfolioEnv was written against the old gym API, but depending on the
# installed version reset() may return (obs, info) and step() may return 5
# values. These helpers normalise both cases transparently.

def _parse_reset(result):
    """Accept both old gym (obs) and new gymnasium (obs, info) reset output."""
    if isinstance(result, tuple):
        return result[0]   # (obs, info) -> obs
    return result          # obs -> obs

def _parse_step(result):
    """
    Accept both:
      old gym      : (obs, reward, done, info)                   - 4 values
      new gymnasium: (obs, reward, terminated, truncated, info)  - 5 values
    Returns (obs, reward, done).
    """
    if len(result) == 5:
        obs, reward, terminated, truncated, _ = result
        return obs, reward, (terminated or truncated)
    obs, reward, done, _ = result
    return obs, reward, done


# ── Policy Network ────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """
    MLP that maps a flattened state to Dirichlet concentration parameters.

    Input  : flat observation vector of size obs_dim (default 21 = 7×3)
    Output : concentration params α ∈ (0, ∞)^{action_dim} via softplus
             (softplus keeps α strictly positive, required by Dirichlet)
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softplus()          # guarantees α > 0 for Dirichlet
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)         # shape: (action_dim,)

    def get_distribution(self, state):
        flat = state.flatten()
        # Sanitize: replace nan/inf that can appear in early env steps
        # (e.g. covariance matrix before enough history, indicator warm-up)
        flat = np.nan_to_num(flat, nan=0.0, posinf=1.0, neginf=-1.0)
        x = torch.FloatTensor(flat)
        alpha = self.forward(x)
        # Clamp for safety: Dirichlet requires strictly positive concentration
        alpha = alpha.clamp(min=1e-6)
        return Dirichlet(alpha)

    def select_action(self, state: np.ndarray):
        """
        Sample portfolio weights from the policy.
        Returns:
            action      : np.ndarray of shape (action_dim,), sums to 1
            log_prob    : scalar tensor (kept in graph for gradient)
        """
        dist = self.get_distribution(state)
        action_tensor = dist.rsample()          # reparameterised sample
        log_prob = dist.log_prob(action_tensor)
        action = action_tensor.detach().numpy()
        return action, log_prob


# ── REINFORCE Agent ───────────────────────────────────────────────────────────

class ReinforceAgent:
    """
    REINFORCE with baseline for continuous portfolio allocation.

    Parameters
    ----------
    obs_dim    : int   — dimension of the flattened observation (default 21)
    action_dim : int   — number of stocks / portfolio weights (default 3)
    hidden_dim : int   — size of hidden layers in the policy network
    lr         : float — learning rate for Adam
    gamma      : float — discount factor
    """

    def __init__(
        self,
        obs_dim: int = 21,
        action_dim: int = 3,
        hidden_dim: int = 64,
        lr: float = 1e-4,
        gamma: float = 0.99,
    ):
        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        self.gamma = gamma
        self.policy = PolicyNetwork(obs_dim, action_dim, hidden_dim)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Buffers filled during one episode, cleared after each update
        self._log_probs: list = []
        self._rewards: list = []

    # ── Interaction helpers ───────────────────────────────────────────────────

    def select_action(self, state: np.ndarray):
        """Select an action and record the log-probability for training."""
        action, log_prob = self.policy.select_action(state)
        self._log_probs.append(log_prob)
        return action

    def record_reward(self, reward: float):
        """Store the (scaled) reward received after an action."""
        self._rewards.append(reward * REWARD_SCALING)

    # ── Core update (called once per episode) ────────────────────────────────

    def update(self):
        """
        Compute the REINFORCE gradient estimate and update the policy.

        Steps:
          1. Compute discounted reward-to-go G_t for each step t.
          2. Subtract the mean G (variance-reduction baseline).
          3. Policy loss = -mean( G_t * log π(a_t | s_t) ).
          4. Backprop and Adam step.
          5. Clear episode buffers.

        Returns the scalar policy loss (float) for logging.
        """
        # 1. Discounted reward-to-go
        T = len(self._rewards)
        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = self._rewards[t] + self.gamma * G
            returns[t] = G

        # 2. Baseline: subtract mean to reduce variance
        returns = returns - returns.mean()

        # Normalise by std if the episode has meaningful variance
        std = returns.std()
        if std > 1e-8:
            returns = returns / std

        returns_tensor = torch.FloatTensor(returns)

        # 3. Policy loss
        log_probs_tensor = torch.stack(self._log_probs)
        loss = -(log_probs_tensor * returns_tensor).mean()

        # 4. Gradient step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 5. Clear buffers
        self._log_probs = []
        self._rewards = []

        return loss.item()

    # ── Full training loop ────────────────────────────────────────────────────

    def train(self, env, n_episodes: int = 500, print_every: int = 10):
        """
        Train the agent on the given environment.

        Parameters
        ----------
        env         : StockPortfolioEnv (old gym API)
        n_episodes  : number of training episodes
        print_every : log interval

        Returns
        -------
        episode_rewards : list of total (unscaled) reward per episode
        episode_losses  : list of policy loss per episode
        """
        episode_rewards = []
        episode_losses = []

        for ep in range(1, n_episodes + 1):
            state = _parse_reset(env.reset())
            done = False
            total_reward = 0.0

            while not done:
                action = self.select_action(state)
                state, reward, done = _parse_step(env.step(action))
                self.record_reward(reward)
                total_reward += reward

            loss = self.update()
            episode_rewards.append(total_reward)
            episode_losses.append(loss)

            if ep % print_every == 0:
                avg_r = np.mean(episode_rewards[-print_every:])
                print(f"Episode {ep:4d}/{n_episodes} | "
                      f"Avg reward (last {print_every}): {avg_r:+.4f} | "
                      f"Loss: {loss:.6f}")

        return episode_rewards, episode_losses

    # ── Evaluation (deterministic — use mean of Dirichlet) ───────────────────

    def evaluate(self, env):
        """
        Run one episode in evaluation mode (no exploration, no gradient).
        Uses the mean of the Dirichlet (α / sum(α)) as a deterministic action.

        Returns
        -------
        total_reward : float
        actions      : list of portfolio weight vectors (one per step)
        """
        self.policy.eval()
        state = _parse_reset(env.reset())
        done = False
        total_reward = 0.0
        actions = []

        with torch.no_grad():
            while not done:
                x = torch.FloatTensor(state.flatten())
                alpha = self.policy(x)
                # Deterministic action: mean of Dirichlet = α / sum(α)
                weights = (alpha / alpha.sum()).numpy()
                actions.append(weights)
                state, reward, done = _parse_step(env.step(weights))
                total_reward += reward

        self.policy.train()
        return total_reward, actions

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        self.policy.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")


# ── Quick smoke-test (run this file directly to verify) ──────────────────────
""""
if __name__ == "__main__":
    # Minimal self-contained test with a dummy env (no FinRL needed)
    import gymnasium as gym

    class DummyPortfolioEnv:
        #Mimics StockPortfolioEnv interface with random data.
        def __init__(self, obs_shape=(7, 3), n_stocks=3, ep_len=252):
            self.obs_shape = obs_shape
            self.n_stocks = n_stocks
            self.ep_len = ep_len
            self._step = 0

        def reset(self):
            self._step = 0
            return np.random.randn(*self.obs_shape).astype(np.float32)

        def step(self, action):
            self._step += 1
            obs = np.random.randn(*self.obs_shape).astype(np.float32)
            reward = float(np.random.randn())
            done = self._step >= self.ep_len
            return obs, reward, done, {}

    print("=== REINFORCE smoke-test ===")
    env = DummyPortfolioEnv()
    agent = ReinforceAgent(obs_dim=21, action_dim=3, hidden_dim=64, lr=1e-3, gamma=0.99)

    rewards, losses = agent.train(env, n_episodes=20, print_every=5)
    test_reward, test_actions = agent.evaluate(env)

    print(f"\nEval reward : {test_reward:.4f}")
    print(f"Sample action (weights): {test_actions[0].round(4)}  — sum={test_actions[0].sum():.4f}")
    print("=== Done ===")
"""


train_env, test_env, _, _ = build_envs()

agent = ReinforceAgent(obs_dim=21, action_dim=3)
agent.train(train_env, n_episodes=200)

reward, actions = agent.evaluate(test_env)

actions = np.array(actions)  # shape: (n_steps, 3)
print(f"Mean weights: AAPL={actions[:,0].mean():.3f}, JPM={actions[:,1].mean():.3f}, XOM={actions[:,2].mean():.3f}")
print(f"Std  weights: AAPL={actions[:,0].std():.3f},  JPM={actions[:,1].std():.3f},  XOM={actions[:,2].std():.3f}")
# %%
"""
Exp3 Agent — Exponential-weight algorithm for Exploration and Exploitation
==========================================================================

Approach : pure adversarial bandit (no state, no MDP).
Each arm  = allocate 100% of the portfolio to one stock.
  arm 0 → [1, 0, 0]  (100% AAPL)
  arm 1 → [0, 1, 0]  (100% JPM)
  arm 2 → [0, 0, 1]  (100% XOM)

Why Exp3 and not UCB / Thompson Sampling ?
  - UCB and Thompson assume a STATIONARY reward distribution.
  - Financial markets are non-stationary (distribution shifts over time).
  - Exp3 makes NO stationarity assumption → better suited to adversarial markets.

Why ignore the state ?
  - Intentional design choice : comparing a stateless bandit (Exp3) against
    state-aware agents (DQN, REINFORCE) directly measures how much the
    state features (covariance + indicators) actually help.

Exp3 update rule (per step) :
  probabilities : p_i = (1-γ) * w_i / Σw  +  γ/K
  reward estimate : r̂_i = r / p_i          (importance-weighted, unbiased)
  weight update   : w_i ← w_i * exp(γ * r̂_i / K)
"""

from __future__ import annotations

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from config import RANDOM_SEED


class Exp3Agent:
    """
    Exp3 adversarial bandit agent for portfolio allocation.

    Parameters
    ----------
    n_arms : int
        Number of arms = number of stocks (default 3).
    gamma : float
        Exploration parameter ∈ (0, 1].
        Higher γ → more exploration, slower exploitation.
        Typical values : 0.05 – 0.2.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, n_arms: int = 3, gamma: float = 0.1, seed: int = RANDOM_SEED):
        self.n_arms = n_arms
        self.gamma  = gamma
        self.rng    = np.random.default_rng(seed)

        # Exp3 weights — initialised to 1 (uniform distribution)
        self.weights = np.ones(n_arms, dtype=np.float64)

        # Memory filled during test()
        self.asset_memory  : list[float] = []
        self.return_memory : list[float] = []
        self.action_memory : list[np.ndarray] = []

    # ── Core Exp3 methods ────────────────────────────────────────────────────

    def _get_probabilities(self) -> np.ndarray:
        """
        Mixed strategy : blend greedy weights with uniform exploration.
            p_i = (1-γ) * w_i / Σw  +  γ/K
        Guarantees p_i > 0 for all arms → importance weights are always finite.
        """
        w_sum = self.weights.sum()
        probs = (1.0 - self.gamma) * (self.weights / w_sum) + self.gamma / self.n_arms
        return probs

    def select_arm(self) -> tuple[int, np.ndarray]:
        """Sample one arm according to current mixed-strategy probabilities."""
        probs = self._get_probabilities()
        arm   = int(self.rng.choice(self.n_arms, p=probs))
        return arm, probs

    def update(self, arm: int, reward: float, probs: np.ndarray) -> None:
        """
        Update the weight of the selected arm.

        Parameters
        ----------
        arm    : index of the arm that was played
        reward : normalised reward ∈ [0, 1]  (see _normalise_reward)
        probs  : probability vector used when arm was selected

        Importance-weighted estimator (unbiased) :
            r̂_arm = reward / p_arm
        Weight update :
            w_arm ← w_arm * exp(γ * r̂_arm / K)

        Note : only the played arm's weight is updated (others unchanged).
        """
        reward_hat          = reward / probs[arm]          # importance weight
        exponent            = self.gamma * reward_hat / self.n_arms
        self.weights[arm]  *= np.exp(exponent)

        # Numerical safety : prevent overflow / underflow
        self.weights = np.clip(self.weights, 1e-300, 1e300)

    # ── Action conversion ────────────────────────────────────────────────────

    def _arm_to_action(self, arm: int) -> np.ndarray:
        """
        Convert a discrete arm index to a continuous portfolio weight vector.
        100% allocated to the selected stock, 0% to others.

        Example : arm=1 → [0., 1., 0.]  (all-in on JPM)

        Note : StockPortfolioEnv passes this through softmax internally,
               so a one-hot vector remains effectively one-hot after softmax
               (one very large logit dominates — we pass raw weights, not logits,
               so the softmax will map [1,0,0] → [~0.58, ~0.21, ~0.21]).

        To get a true 100% allocation, pass large logits instead :
        """
        action = np.zeros(self.n_arms, dtype=np.float32)
        action[arm] = 10.0   # large logit → softmax ≈ [1, 0, 0]
        return action

    @staticmethod
    def _normalise_reward(new_value: float, old_value: float) -> float:
        """
        Convert raw portfolio value to a normalised reward ∈ [0, 1].

        We use daily portfolio return, clipped to a realistic range [-5%, +5%]
        and linearly mapped to [0, 1] :
            normalised = (return + 0.05) / 0.10

        This keeps the Exp3 importance-weighted estimator numerically stable
        and aligns with the theoretical assumption r ∈ [0, 1].
        """
        daily_return = (new_value - old_value) / old_value  # e.g. +0.003
        normalised   = (daily_return + 0.05) / 0.10         # map [-5%,+5%] → [0,1]
        return float(np.clip(normalised, 0.0, 1.0))

    # ── Training loop ────────────────────────────────────────────────────────

    def train(self, env, n_episodes: int = 5) -> None:
        """
        Run Exp3 on the training environment for n_episodes full passes.

        Each episode = one full pass through the training period.
        Weights are NOT reset between episodes so the agent accumulates
        knowledge across all passes.

        Parameters
        ----------
        env        : StockPortfolioEnv  (training set)
        n_episodes : number of full passes through the training data
        """
        self.weights = np.ones(self.n_arms, dtype=np.float64)  # fresh start

        for ep in range(n_episodes):
            state, _  = env.reset()
            done       = False
            prev_value = float(env.portfolio_value)

            while not done:
                arm, probs = self.select_arm()
                action     = self._arm_to_action(arm)

                next_state, reward, terminated, truncated, _ = env.step(action)
                done       = terminated or truncated
                new_value  = float(reward)

                # Normalise reward before updating weights
                norm_r = self._normalise_reward(new_value, prev_value)
                self.update(arm, norm_r, probs)

                prev_value = new_value
                state      = next_state

            print(f"[Exp3] Episode {ep + 1}/{n_episodes} "
                  f"— final portfolio value : {new_value:,.2f} "
                  f"— weights : {self.weights.round(4)}")

    # ── Evaluation ──────────────────────────────────────────────────────────

    def test(self, env) -> dict:
        """
        Evaluate the trained Exp3 agent on the test environment.
        Weights are frozen (no update during test).

        Returns
        -------
        dict with keys : portfolio_values, daily_returns, actions,
                         total_return, sharpe
        """
        self.asset_memory  = []
        self.return_memory = []
        self.action_memory = []

        state, _       = env.reset()
        done            = False
        portfolio_value = float(env.portfolio_value)
        self.asset_memory.append(portfolio_value)

        while not done:
            arm, _  = self.select_arm()          # no update during test
            action   = self._arm_to_action(arm)
            self.action_memory.append(action.copy())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done  = terminated or truncated
            new_v = float(reward)

            daily_ret = (new_v - portfolio_value) / portfolio_value
            self.return_memory.append(daily_ret)
            self.asset_memory.append(new_v)
            portfolio_value = new_v
            state = next_state

        return self.get_results()

    # ── Metrics & results ────────────────────────────────────────────────────

    def _compute_metrics(self) -> tuple[float, float]:
        """
        Compute total return and annualised Sharpe ratio.

        Sharpe = sqrt(252) * mean(daily_returns) / std(daily_returns)
        """
        returns      = np.array(self.return_memory)
        total_return = (self.asset_memory[-1] - self.asset_memory[0]) / self.asset_memory[0]

        if returns.std() > 0:
            sharpe = float((252 ** 0.5) * returns.mean() / returns.std())
        else:
            sharpe = 0.0

        return float(total_return), sharpe

    def get_results(self) -> dict:
        """
        Return a results dict compatible with run_comparison.py.

        Keys
        ----
        portfolio_values : list of portfolio value at each step
        daily_returns    : list of daily returns
        actions          : list of action vectors
        total_return     : float, e.g. 0.15 means +15%
        sharpe           : float, annualised Sharpe ratio
        """
        total_return, sharpe = self._compute_metrics()
        return {
            "portfolio_values" : self.asset_memory,
            "daily_returns"    : self.return_memory,
            "actions"          : self.action_memory,
            "total_return"     : total_return,
            "sharpe"           : sharpe,
        }








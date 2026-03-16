from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

try:
    from stable_baselines3 import DDPG
    from stable_baselines3.common.noise import NormalActionNoise
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False


class FlattenObsWrapper(gym.ObservationWrapper):
    """Flatten 2D observation (7x3) to 1D (21) for SB3 compatibility."""
    def __init__(self, env):
        super().__init__(env)
        shape = env.observation_space.shape
        flat_dim = int(np.prod(shape))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )

    def observation(self, obs):
        return np.asarray(obs, dtype=np.float32).flatten()


def _build_results(env) -> dict:
    """Build standardized results dict from env memory."""
    returns = np.array(env.portfolio_return_memory)
    values  = env.asset_memory
    total_return = (values[-1] - values[0]) / values[0]
    sharpe = float((252**0.5) * returns.mean() / returns.std()) if returns.std() > 1e-9 else 0.0
    return {
        "portfolio_values": list(values),
        "daily_returns":    list(returns),
        "total_return":     float(total_return),
        "sharpe":           sharpe,
    }


class BuyAndHoldBaseline:
    """Equal-weight buy-and-hold baseline: 1/3 AAPL, 1/3 JPM, 1/3 XOM."""

    def test(self, env) -> dict:
        state, _ = env.reset()
        done = False
        weights = np.ones(env.stock_dim, dtype=np.float32) / env.stock_dim
        while not done:
            _, _, terminated, truncated, _ = env.step(weights)
            done = terminated or truncated
        return _build_results(env)


class DDPGBaseline:
    """DDPG baseline using Stable-Baselines3."""

    def __init__(self):
        if not _SB3_AVAILABLE:
            raise ImportError("stable_baselines3 is required for DDPGBaseline.")
        self.model = None

    def train(self, env, timesteps: int = 20_000) -> None:
        wrapped = FlattenObsWrapper(env)
        n_actions = wrapped.action_space.shape[0]
        noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.1 * np.ones(n_actions),
        )
        self.model = DDPG(
            "MlpPolicy",
            wrapped,
            action_noise=noise,
            verbose=0,
            seed=42,
        )
        self.model.learn(total_timesteps=timesteps)

    def test(self, env) -> dict:
        if self.model is None:
            raise RuntimeError("Call train() before test().")
        wrapped = FlattenObsWrapper(env)
        obs, _ = wrapped.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = wrapped.step(action)
            done = terminated or truncated
        return _build_results(env)

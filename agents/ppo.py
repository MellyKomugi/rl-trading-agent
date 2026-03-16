from __future__ import annotations

import copy

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from .architectures import EIIE
from .utils import PVM


class PPO:
    def __init__(
        self,
        env,
        policy=EIIE,
        policy_kwargs=None,
        validation_env=None,
        batch_size=100,
        lr=1e-3,
        action_noise=0.0,
        optimizer=AdamW,
        device="cpu",
        gamma=0.99,
        clip_epsilon=0.2,
        update_epochs=5,
        dirichlet_scale=50.0,
        verbose=1,
        seed=None,
    ):
        self.policy = policy
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.validation_env = validation_env
        self.batch_size = batch_size
        self.lr = lr
        self.action_noise = action_noise
        self.optimizer = optimizer
        self.device = device

        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.dirichlet_scale = dirichlet_scale

        self.verbose = verbose
        self.seed = seed

        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self._setup_train(env, self.policy, self.lr, self.optimizer)


    @staticmethod
    def _get_episode_length(env):
        if hasattr(env, "episode_length"):
            return env.episode_length
        if hasattr(env, "df"):
            return len(env.df.index.unique()) - 1
        raise AttributeError(
            "Environment must define `episode_length` or expose a dataframe in `df`."
        )

    @staticmethod
    def _get_portfolio_size(env):
        if hasattr(env, "portfolio_size"):
            return env.portfolio_size
        if hasattr(env, "stock_dim"):
            return env.stock_dim
        if hasattr(env, "action_space") and getattr(env.action_space, "shape", None):
            return env.action_space.shape[0]
        raise AttributeError(
            "Environment must define `portfolio_size`, `stock_dim`, or an action space shape."
        )

    @staticmethod
    def _extract_obs(reset_output):
        if isinstance(reset_output, tuple):
            return reset_output[0]
        return reset_output

    @staticmethod
    def _unpack_step_output(step_output):
        if len(step_output) == 5:
            next_obs, reward, terminated, truncated, info = step_output
            done = terminated or truncated
            return next_obs, reward, done, info
        next_obs, reward, done, info = step_output
        return next_obs, reward, done, info

    def _setup_train(self, env, policy, lr, optimizer):
        self.train_env = env
        self.train_policy = policy(**self.policy_kwargs).to(self.device)
        self.train_optimizer = optimizer(self.train_policy.parameters(), lr=lr)

        self.train_pvm = PVM(
            self._get_episode_length(self.train_env),
            self._get_portfolio_size(self.train_env),
        )

    def _setup_test(self, env, policy, lr, optimizer):
        self.test_env = env

        policy = self.train_policy if policy is None else policy
        lr = self.lr if lr is None else lr
        optimizer = self.optimizer if optimizer is None else optimizer

        self.test_policy = copy.deepcopy(policy).to(self.device)
        self.test_optimizer = optimizer(self.test_policy.parameters(), lr=lr)

        self.test_pvm = PVM(
            self._get_episode_length(self.test_env),
            self._get_portfolio_size(self.test_env),
        )

    def _to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=torch.float32)
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _get_dist(self, policy, obs, last_action):
        mu = policy.mu(obs, last_action)
        alpha = mu * self.dirichlet_scale + 1e-3
        return torch.distributions.Dirichlet(alpha)

    def _compute_discounted_returns(self, rewards, dones):
        returns = []
        discounted = 0.0

        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted = 0.0
            discounted = reward + self.gamma * discounted
            returns.append(discounted)

        returns.reverse()
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        if returns.numel() <= 1:
            return returns

        std = returns.std(unbiased=False)
        if torch.isnan(std) or std.item() < 1e-8:
            return returns - returns.mean()

        return (returns - returns.mean()) / (std + 1e-8)


    def _collect_trajectory(self, env, policy, pvm):
        obs = self._extract_obs(env.reset())
        pvm.reset()
        done = False

        trajectory = {
            "obs": [],
            "last_actions": [],
            "actions": [],
            "log_probs": [],
            "rewards": [],
            "dones": [],
        }

        episode_reward = 0.0
        final_value = None
        step_count = 0

        while not done:
            last_action = pvm.retrieve()

            obs_t = self._to_tensor(obs).unsqueeze(0)
            last_action_t = self._to_tensor(last_action).unsqueeze(0)

            dist = self._get_dist(policy, obs_t, last_action_t)
            action_t = dist.sample()
            log_prob_t = dist.log_prob(action_t)

            action = action_t.squeeze(0).detach().cpu().numpy()
            pvm.add(action)

            next_obs, _, done, info = self._unpack_step_output(env.step(action))

            price_variation = np.asarray(info["price_variation"], dtype=np.float32)
            trf_mu = float(info["trf_mu"])
            gross_return = float(np.sum(action * price_variation) * trf_mu)
            reward = float(np.log(max(gross_return, 1e-8)))

            trajectory["obs"].append(np.asarray(obs, dtype=np.float32))
            trajectory["last_actions"].append(np.asarray(last_action, dtype=np.float32))
            trajectory["actions"].append(np.asarray(action, dtype=np.float32))
            trajectory["log_probs"].append(float(log_prob_t.item()))
            trajectory["rewards"].append(reward)
            trajectory["dones"].append(done)

            obs = next_obs
            episode_reward += reward
            step_count += 1

            if getattr(env, "asset_memory", None):
                final_value = env.asset_memory[-1]

        return trajectory, episode_reward, final_value, step_count

    def _ppo_update(self, trajectory, test=False):
        policy = self.test_policy if test else self.train_policy
        optimizer = self.test_optimizer if test else self.train_optimizer

        obs = self._to_tensor(np.asarray(trajectory["obs"], dtype=np.float32))
        last_actions = self._to_tensor(
            np.asarray(trajectory["last_actions"], dtype=np.float32)
        )
        actions = self._to_tensor(np.asarray(trajectory["actions"], dtype=np.float32))
        old_log_probs = self._to_tensor(
            np.asarray(trajectory["log_probs"], dtype=np.float32)
        )
        returns = self._compute_discounted_returns(
            trajectory["rewards"], trajectory["dones"]
        )

        last_loss = None
        n_samples = obs.shape[0]

        for _ in range(self.update_epochs):
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size

                obs_batch = obs[start:end]
                last_actions_batch = last_actions[start:end]
                actions_batch = actions[start:end]
                old_log_probs_batch = old_log_probs[start:end]
                returns_batch = returns[start:end]

                dist = self._get_dist(policy, obs_batch, last_actions_batch)
                new_log_probs = dist.log_prob(actions_batch)

                ratio = torch.exp(new_log_probs - old_log_probs_batch)
                unclipped = ratio * returns_batch
                clipped = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_epsilon,
                        1.0 + self.clip_epsilon,
                    )
                    * returns_batch
                )

                loss = -torch.mean(torch.min(unclipped, clipped))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                last_loss = loss.item()

        return last_loss

    def train(self, episodes=100):
        pbar = tqdm(range(1, episodes + 1), desc="Training PPO")
        last_loss = None

        for episode in pbar:
            trajectory, episode_reward, final_value, step_count = self._collect_trajectory(
                self.train_env, self.train_policy, self.train_pvm
            )

            last_loss = self._ppo_update(trajectory, test=False)

            if self.validation_env:
                self.test(self.validation_env)

            pbar.set_postfix(
                {
                    "reward": f"{episode_reward:.4f}",
                    "loss": f"{last_loss:.6f}" if last_loss is not None else "NA",
                    "pv": f"{final_value:.2f}" if final_value is not None else "NA",
                    "steps": step_count,
                }
            )

    def test(
        self, env, policy=None, online_training_period=10, lr=None, optimizer=None
    ):
        """Run evaluation on the test environment without online updates."""
        self._setup_test(env, policy, lr, optimizer)

        obs = self._extract_obs(self.test_env.reset())
        self.test_pvm.reset()
        done = False

        while not done:
            last_action = self.test_pvm.retrieve()

            obs_t = self._to_tensor(obs).unsqueeze(0)
            last_action_t = self._to_tensor(last_action).unsqueeze(0)

            # deterministic portfolio weights
            action_t = self.test_policy.mu(obs_t, last_action_t)
            action = action_t.squeeze(0).detach().cpu().numpy()

            self.test_pvm.add(action)

            next_obs, _, done, info = self._unpack_step_output(self.test_env.step(action))
            obs = next_obs

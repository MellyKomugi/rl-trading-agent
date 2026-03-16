from __future__ import annotations

import pandas as pd
# from agents.algorithms import PPO
from agents.algorithms import PolicyGradient
from agents.architectures import SimplePortfolioMLP
# from stable_baselines3.common.vec_env import VecEnv

# MODELS = {"ppo": PPO , "pg": PolicyGradient}
MODELS = {"pg": PolicyGradient}
architecture={"MlpPolicy" : SimplePortfolioMLP}

class DRLAgent:
    """Generic facade for portfolio DRL models.

    The current implementation supports PPO via Stable-Baselines3 and keeps
    a model registry/API that can later host custom portfolio algorithms.
    """

    def __init__(self, env):
        self.env = env

    def _resolve_training_env(self, model_name):
        """Return the env format expected by the requested model."""
        if model_name == "ppo":
            # if isinstance(self.env, VecEnv):
            #     return self.env
            if hasattr(self.env, "get_sb_env"):
                vec_env, _ = self.env.get_sb_env()
                return vec_env

        return self.env

    @staticmethod
    def _extract_raw_env(environment):
        if hasattr(environment, "get_sb_env"):
            return environment
        if hasattr(environment, "envs") and environment.envs:
            return environment.envs[0]
        raise ValueError("Unable to resolve the raw portfolio environment.")

    @staticmethod
    def _build_output_frames(raw_env):
        results_df = pd.DataFrame(
            {
                "date": pd.to_datetime(raw_env.date_memory),
                "portfolio_value": raw_env.asset_memory,
                "daily_return": raw_env.portfolio_return_memory,
            }
        )

        actions_df = raw_env.save_action_memory().reset_index()
        if "index" in actions_df.columns and "date" not in actions_df.columns:
            actions_df = actions_df.rename(columns={"index": "date"})
        if "date" in actions_df.columns:
            actions_df["date"] = pd.to_datetime(actions_df["date"])

        return results_df, actions_df

    def get_model(
        self,
        model_name,
        policy="MlpPolicy",
        device="cpu",
        model_kwargs=None,
        policy_kwargs=None,
        verbose=1,
        seed=None,
    ):
        """Instantiate a registered DRL model for portfolio optimization."""
        if model_name not in MODELS:
            raise NotImplementedError(
                f"Model '{model_name}' is not registered in MODELS."
            )

        model_cls = MODELS[model_name]
        model_kwargs = {} if model_kwargs is None else model_kwargs.copy()
        policy_kwargs = {} if policy_kwargs is None else policy_kwargs.copy()
        env = self._resolve_training_env(model_name)

        if model_name == "ppo":
            if device is not None:
                model_kwargs["device"] = device
            if policy_kwargs:
                model_kwargs["policy_kwargs"] = policy_kwargs
            return model_cls(
                policy=policy,
                env=env,
                verbose=verbose,
                seed=seed,
                **model_kwargs,
            )
        if model_name =="pg" : 
            if device is not None : 
                model_kwargs['device']=device
            if policy_kwargs : 
                model_kwargs["policy_kwargs"]=policy_kwargs
            return model_cls(
                env=env, 
                policy=architecture.get(policy, policy), 
                **model_kwargs
            )                

        return model_cls(env, **model_kwargs)

    @staticmethod
    def train_model(model, episodes=None, total_timesteps=None):
        """Train either an SB3 model or a future custom portfolio model."""
        if hasattr(model, "learn"):
            timesteps = 5000 if total_timesteps is None else total_timesteps
            model.learn(total_timesteps=timesteps)
            return model

        if hasattr(model, "train"):
            num_episodes = 100 if episodes is None else episodes
            model.train(num_episodes)
            return model

        raise TypeError("Unsupported model type: expected a learn() or train() method.")

    @staticmethod
    def DRL_validation(model, test_env, **kwargs):
        """Run validation and return the standard portfolio outputs."""
        return DRLAgent.DRL_prediction(model, test_env, **kwargs)

    @staticmethod
    def DRL_prediction(model, environment, deterministic=True, **kwargs):
        """Run inference on the portfolio env and return results/actions frames."""
        raw_env = DRLAgent._extract_raw_env(environment)

        if hasattr(model, "predict"):
            if hasattr(environment, "get_sb_env"):
                test_env, test_obs = environment.get_sb_env()
            else:
                test_env = environment
                test_obs = test_env.reset()

            test_obs = test_env.reset()
            max_steps = len(raw_env.df.index.unique())

            for _ in range(max_steps):
                action, _ = model.predict(test_obs, deterministic=deterministic)
                step_output = test_env.step(action)

                if len(step_output) == 4:
                    test_obs, _, dones, _ = step_output
                else:
                    test_obs, _, dones, _, _ = step_output

                if dones[0]:
                    break

            return DRLAgent._build_output_frames(raw_env)

        if hasattr(model, "test"):
            model.test(environment, **kwargs)
            return DRLAgent._build_output_frames(raw_env)

        raise TypeError("Unsupported model type: expected a predict() or test() method.")

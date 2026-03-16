from __future__ import annotations

from importlib import import_module
from pathlib import Path

import pandas as pd

from agents.DRL import DRLAgent
from agents.architectures import SimplePortfolioMLP
from environment.setup_env import build_envs


def _load_optional_class(module_path, class_name):
    try:
        module = import_module(module_path)
    except ImportError as exc:
        print(f"[skip] {class_name}: import failed ({exc})")
        return None

    cls = getattr(module, class_name, None)
    if cls is None:
        print(f"[skip] {class_name}: class not found in {module_path}")
    return cls


def _compute_summary(model_name, reward_type, results_df):
    initial_value = float(results_df["portfolio_value"].iloc[0])
    final_value = float(results_df["portfolio_value"].iloc[-1])
    total_return = (final_value / initial_value) - 1 if initial_value else 0.0
    daily_returns = results_df["daily_return"].astype(float)
    sharpe = 0.0
    if daily_returns.std() != 0:
        sharpe = float((252**0.5) * daily_returns.mean() / daily_returns.std())

    return {
        "model": model_name,
        "reward_type": reward_type,
        "initial_portfolio_value": initial_value,
        "final_portfolio_value": final_value,
        "total_return": total_return,
        "sharpe": sharpe,
        "num_steps": len(results_df) - 1,
    }


def _save_model_outputs(model_name, reward_type, results_df, actions_df, output_dir):
    suffix = f"{model_name}_{reward_type}"
    results_path = output_dir / f"results_{suffix}.csv"
    actions_path = output_dir / f"actions_{suffix}.csv"
    results_df.to_csv(results_path, index=False)
    actions_df.to_csv(actions_path, index=False)
    print(f"[saved] {model_name} results -> {results_path}")
    print(f"[saved] {model_name} actions -> {actions_path}")
    return results_path, actions_path


def _print_test_results(summary_df):
    print("\n[test results]")
    print(summary_df.to_string(index=False))


def _run_policy_gradient(train_env, test_env, device="cpu", episodes=50):
    agent = DRLAgent(env=train_env)
    pg_model = agent.get_model(
        "pg",
        policy=SimplePortfolioMLP,
        device=device,
        model_kwargs={
            "batch_size": 64,
            "lr": 1e-3,
            "action_noise": 0.01,
        },
        policy_kwargs={
            "input_shape": (7, 3),
            "portfolio_size": train_env.stock_dim,
            "hidden_dim": 64,
            "device": device,
        },
    )
    trained_pg = agent.train_model(pg_model, episodes=episodes)
    return DRLAgent.DRL_prediction(
        model=trained_pg,
        environment=test_env,
        online_training_period=10**9,
        lr=0,
    )


def _run_ppo(train_env, test_env, device="cpu", episodes=50):
    agent = DRLAgent(env=train_env)
    ppo_model = agent.get_model(
        "ppo",
        policy=SimplePortfolioMLP,
        device=device,
        model_kwargs={
            "batch_size": 64,
            "lr": 1e-3,
            "action_noise": 0.01,
            "gamma": 0.99,
            "clip_epsilon": 0.2,
            "update_epochs": 5,
            "dirichlet_scale": 50.0,
        },
        policy_kwargs={
            "input_shape": (7, 3),
            "portfolio_size": train_env.stock_dim,
            "hidden_dim": 64,
            "device": device,
        },
    )
    trained_ppo = agent.train_model(ppo_model, episodes=episodes)
    return DRLAgent.DRL_prediction(
        model=trained_ppo,
        environment=test_env,
        online_training_period=10**9,
        lr=0,
    )


def _run_exp3(train_env, test_env, train_episodes=5):
    exp3_cls = _load_optional_class("agents.exp3", "Exp3Agent")
    if exp3_cls is None:
        return None

    agent = exp3_cls(n_arms=train_env.stock_dim)
    agent.train(train_env, n_episodes=train_episodes)
    agent.test(test_env)
    return DRLAgent._build_output_frames(test_env)


def _run_reinforce(train_env, test_env, train_episodes=50):
    from agents.reinforce import ReinforceAgent
    agent = ReinforceAgent(obs_dim=21, action_dim=3)
    agent.train(train_env, n_episodes=train_episodes)
    agent.evaluate(test_env)
    return DRLAgent._build_output_frames(test_env)


def run_comparison(
    if_using_exp3: bool = True,
    if_using_reinforce: bool = True,
    if_using_dqn: bool = False,
    if_using_policy_gradient: bool = False,
    if_using_ppo: bool = False,
    reward_type: str = "portfolio_value",
    ppo_train_episodes: int = 10,
    policy_gradient_episodes: int = 10,
    exp3_train_episodes: int = 10,
    reinforce_train_episodes: int = 10,
    print_test_results: bool = True,
    output_dir: str | None = None,
):
    del if_using_dqn

    train_env, test_env, train_df, test_df = build_envs(reward_type=reward_type)
    print(type(train_env))

    base_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).resolve().parent / "outputs" / reward_type
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    results_by_model = {}
    actions_by_model = {}
    summaries = []
    comparison_frames = []

    if if_using_policy_gradient:
        print("[run] PolicyGradient")
        print(f"[train] PolicyGradient episodes={policy_gradient_episodes}")
        results_pg, actions_pg = _run_policy_gradient(
            train_env, test_env, episodes=policy_gradient_episodes
        )
        results_by_model["policy_gradient"] = results_pg
        actions_by_model["policy_gradient"] = actions_pg

    if if_using_ppo:
        print("[run] PPO")
        print(f"[train] PPO episodes={ppo_train_episodes}")
        results_ppo, actions_ppo = _run_ppo(
            train_env, test_env, episodes=ppo_train_episodes
        )
        results_by_model["ppo"] = results_ppo
        actions_by_model["ppo"] = actions_ppo

    if if_using_exp3:
        print("[run] Exp3")
        print(f"[train] Exp3 episodes={exp3_train_episodes}")
        exp3_outputs = _run_exp3(
            train_env, test_env, train_episodes=exp3_train_episodes
        )
        if exp3_outputs is not None:
            results_exp3, actions_exp3 = exp3_outputs
            results_by_model["exp3"] = results_exp3
            actions_by_model["exp3"] = actions_exp3

    if if_using_reinforce:
        print("[run] Reinforce")
        print(f"[train] Reinforce episodes={reinforce_train_episodes}")
        reinforce_outputs = _run_reinforce(
            train_env, test_env, train_episodes=reinforce_train_episodes
        )
        if reinforce_outputs is not None:
            results_reinforce, actions_reinforce = reinforce_outputs
            results_by_model["reinforce"] = results_reinforce
            actions_by_model["reinforce"] = actions_reinforce

    if not results_by_model:
        raise RuntimeError("No model was executed successfully.")

    for model_name, results_df in results_by_model.items():
        actions_df = actions_by_model[model_name]
        _save_model_outputs(model_name, reward_type, results_df, actions_df, base_output_dir)

        model_results = results_df.copy()
        model_results["model"] = model_name
        model_results["reward_type"] = reward_type
        comparison_frames.append(model_results)
        summaries.append(_compute_summary(model_name, reward_type, results_df))

    comparison_df = pd.concat(comparison_frames, ignore_index=True)
    summary_df = pd.DataFrame(summaries).sort_values(
        by="final_portfolio_value", ascending=False
    )

    comparison_path = base_output_dir / f"comparison_timeseries_{reward_type}.csv"
    summary_path = base_output_dir / f"comparison_summary_{reward_type}.csv"
    comparison_df.to_csv(comparison_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"[saved] comparison timeseries -> {comparison_path}")
    print(f"[saved] comparison summary -> {summary_path}")
    if print_test_results:
        _print_test_results(summary_df)

    return {
        "reward_type": reward_type,
        "output_dir": str(base_output_dir),
        "train_df": train_df,
        "test_df": test_df,
        "results": results_by_model,
        "actions": actions_by_model,
        "comparison": comparison_df,
        "summary": summary_df,
    }






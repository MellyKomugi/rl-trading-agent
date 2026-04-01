# rl-trading-agent

This project explores the use of Reinforcement Learning (RL) for portfolio allocation.
We implement and compare several agents in a financial environment inspired by FinRL,
including REINFORCE (Policy Gradient), PPO, EXP3 (adversarial bandit), and baselines
(Buy & Hold, DDPG).

The final training, evaluation, and visualization are in `notebook.ipynb`.

---

## Project Objective

The goal is to study how different reinforcement learning algorithms perform on a
portfolio allocation task using three stocks (AAPL, JPM, XOM).

We focus on:
- Learning dynamic asset allocation policies
- Evaluating the impact of different reward functions (ablation study)
- Comparing RL agents against simple baselines

---

## Project Structure

```
rl-trading-agent/
|
├── agents/                # RL algorithms
│   ├── exp3.py            # EXP3 adversarial bandit
│   ├── reinforce.py       # REINFORCE (Policy Gradient)
│   ├── ppo.py             # PPO (custom implementation)
│   ├── architecture.py    # Neural network architectures (MLP, EIIE)
│   └── utils.py           # PVM, ReplayBuffer
|
├── application/           # Experiment pipelines
│   └── run_comparison.py  # Unified training & evaluation pipeline
|
├── baselines/
│   └── baselines.py       # Buy & Hold, DDPG (SB3)
|
├── environment/
│   ├── portfolio_env.py   # Custom FinRL-style trading environment
│   ├── setup_env.py       # Environment builder
│   └── preprocessors.py  # Feature engineering
|
├── notebook.ipynb         # Final submission notebook (training + results)
├── config.py              # Global config (stocks, dates, hyperparameters)
├── requirements.txt       # Dependencies
└── README.md
```

---

## Environment

Custom portfolio allocation environment inspired by FinRL.

Key characteristics:
- State: 7x3 matrix — covariance features + technical indicators (MACD, RSI, CCI, DX)
- Actions: portfolio weights (continuous, normalized via softmax)
- Reward: configurable — portfolio_value, portfolio_return, or log_return
- Transaction costs: 0.1% per trade
- Each step = one trading day

---

## Implemented Agents

| Agent      | Type               | Description                                                       |
|------------|--------------------|-------------------------------------------------------------------|
| REINFORCE  | Policy Gradient    | MLP policy, Monte Carlo returns, variance-reduction baseline      |
| PPO        | Actor-Critic       | Clipped surrogate objective, Portfolio Vector Memory              |
| EXP3       | Adversarial Bandit | Importance-weighted updates, gamma=0.1, regret O(sqrt(KT log K)) |
| Buy & Hold | Baseline           | Equal weights (1/3 each stock), no rebalancing                    |
| DDPG       | Baseline           | SB3 implementation with flattened observation space               |

---

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Open and run the main notebook:

```bash
jupyter notebook notebook.ipynb
```

3. The notebook will:
   - Download and preprocess data (AAPL, JPM, XOM)
   - Train all agents
   - Evaluate on the test set
   - Generate comparison plots and the reward ablation study

Set `QUICK_TEST = True` at the top of the notebook for a fast smoke-test run.

---

## Data

- Assets: AAPL, JPM, XOM
- Train period: 2019-2021 (bull market)
- Test period: 2022-2023 (bear market)

The regime change between train and test sets provides a realistic out-of-sample evaluation.
Data is downloaded automatically via yfinance at runtime — no manual download required.

---

## Evaluation

Agents are evaluated on:
- Total return (test period)
- Sharpe ratio (annualised)
- Portfolio value over time

A reward ablation study compares all agents across three reward functions:
portfolio_value, portfolio_return, and log_return.

---

## Authors

Project developed as part of the Reinforcement Learning and Autonomous Agents course
at Ecole Polytechnique (2025-2026).

Wandrille BOIVIN, Sylvain DEHAYEM, Antoine DELACOUR, Mouhamadou Moustapha DIOP

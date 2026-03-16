# Purpose : download market data, compute features and covariance,
#           split into train/test sets, and create the FinRL portfolio env.
#
# Design choices :
#   - 3 stocks from different sectors (tech, finance, energy) to make
#     portfolio allocation meaningful (low cross-correlation)
#   - 252-day rolling covariance = 1 trading year, standard in finance
#   - Train 2019-2021 (bull market) / Test 2022-2023 (bear market)
#     → tests whether agents generalise to unseen market regimes

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer
from finrl.meta.env_portfolio_allocation.env_portfolio import StockPortfolioEnv
from config import (
    STOCKS, TRAIN_START, TEST_START, TEST_END,
    INITIAL_CAPITAL, TRANSACTION_COST, INDICATORS
)


# ── Step 1 ── Download historical prices from Yahoo Finance ──────────────────

def get_data():
    df = YahooDownloader(
        start_date=TRAIN_START,
        end_date=TEST_END,
        ticker_list=STOCKS
    ).fetch_data()
    return df


# ── Step 2 ── Add technical indicators ───────────────────────────────────────
# FinRL uses stockstats to compute MACD, RSI, CCI, DX for each stock.
# NaN values at the start of the series (indicators need history) are
# forward-filled then zero-filled.

def add_features(df):
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=INDICATORS,
        use_turbulence=False,
        user_defined_feature=False
    )
    df = fe.preprocess_data(df)
    df = df.ffill().fillna(0)
    return df


# ── Step 3 ── Compute rolling covariance matrix ───────────────────────────────
# StockPortfolioEnv requires a 'cov_list' column (3×3 covariance matrix
# per date). We compute it over a 252-day rolling window (1 trading year).
# The first 252 days are dropped since there is not enough history.

def add_covariance(df, lookback=252):
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    df.index = df.date.factorize()[0]

    unique_dates = df.date.unique()
    cov_list = []

    for i in range(lookback, len(unique_dates)):
        prices = df.loc[i - lookback:i].pivot_table(
            index='date', columns='tic', values='close'
        )
        returns = prices.pct_change().dropna()
        cov_list.append(returns.cov().values)

    df_cov = pd.DataFrame({
        'date': unique_dates[lookback:],
        'cov_list': cov_list
    })

    df = df.merge(df_cov, on='date')
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    return df


# ── Step 4 ── Split into train and test sets ──────────────────────────────────
# Train : 2019-2021 (bull market, post-lookback period)
# Test  : 2022-2023 (bear market, out-of-sample)
# reset_index is required so that StockPortfolioEnv can access rows via loc[day].

def split_data(df):
    train = df[df.date < TEST_START].reset_index(drop=True)
    test  = df[df.date >= TEST_START].reset_index(drop=True)
    print(f"Train : {train.date.min()} → {train.date.max()} ({len(train)} rows)")
    print(f"Test  : {test.date.min()} → {test.date.max()} ({len(test)} rows)")
    return train, test


# ── Step 5 ── Create the FinRL portfolio environment ─────────────────────────
# StockPortfolioEnv :
#   - action  = portfolio weights [w_AAPL, w_JPM, w_XOM], passed through softmax
#   - state   = rolling covariance matrix + technical indicators
#   - reward  = raw portfolio value (agents will apply REWARD_SCALING themselves)
# Note : hmax, transaction_cost_pct and reward_scaling are required by __init__
#        but are not used internally by this environment version.

def make_env(df):
    stock_dim = len(STOCKS)

    # StockPortfolioEnv expects rows indexed by day number (not row number)
    df = df.sort_values(['date', 'tic']).reset_index(drop=True)
    df.index = df.date.factorize()[0]

    env = StockPortfolioEnv(
        df=df,
        stock_dim=stock_dim,
        hmax=100,
        initial_amount=INITIAL_CAPITAL,
        transaction_cost_pct=TRANSACTION_COST,
        reward_scaling=1e-4,
        state_space=stock_dim,        # width of the observation matrix
        action_space=stock_dim,       # one weight per stock
        tech_indicator_list=INDICATORS,
        turbulence_threshold=None,    # no forced sell-off
    )
    return env


# ── Main pipeline ─────────────────────────────────────────────────────────────

def build_envs():
    # FinRL saves plots to results/ at episode end, create silently
    os.makedirs("results", exist_ok=True)

    print("1. Downloading data")
    df = get_data()

    print("2. Adding technical indicators")
    df = add_features(df)

    print("3. Computing rolling covariance")
    df = add_covariance(df)

    print("4. Splitting train / test")
    train_df, test_df = split_data(df)

    print("5. Creating environments")
    train_env = make_env(train_df)
    test_env  = make_env(test_df)

    print("Done")
    return train_env, test_env, train_df, test_df


# Stocks 
STOCKS = ["AAPL", "JPM", "XOM"]

# Periods
TRAIN_START = "2018-01-01"
TRAIN_END   = "2021-12-31"
TEST_START  = "2022-01-01"
TEST_END    = "2023-12-31"

# Portfolio
INITIAL_CAPITAL = 100_000
TRANSACTION_COST = 0.001  # 0.1%

REWARD_SCALING = 1e-4 # Intended to divide the reward by 10,000 to stabilize training


# technical features 
INDICATORS = ["macd", "rsi_30", "cci_30", "dx_30"]    
                                                                   
# Hyperparameters
RANDOM_SEED = 42

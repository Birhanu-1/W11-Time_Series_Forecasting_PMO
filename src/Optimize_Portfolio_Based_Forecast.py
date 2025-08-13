# ===============================
1. Prepare Expected Returns
TSLA:

Take the annualized return from your Task 2 best model forecast.

If your model outputs daily forecasted prices, first calculate daily % changes, then annualize:

𝑅
annual
=(1+𝑅ˉdaily)252−1R 

annual=(1+Rˉdaily)252−1
BND & SPY:

Use historical mean daily returns (July 2015 – July 2025).

Annualize them with the same formula above.

2. Compute Covariance Matrix
Use historical daily returns of TSLA, BND, SPY (same date range).

Convert to NumPy array and use:

cov_matrix = returns.cov() * 252  # annualized covariance
This matrix will be used in the risk calculation.

# ============================================
# Implementation (NumPy + SciPy)
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# If you have LSTM forecast available:
tsla_forecast_prices = lstm_fc  # Make sure lstm_fc is defined
# Replace with your actual forecasted prices
tsla_forecast_prices = arima_fc  # or lstm_fc, whichever you want to use

# Calculate daily returns
tsla_forecast_daily_returns = tsla_forecast_prices.pct_change().dropna()

# Mean daily return
tsla_mean_daily_return = tsla_forecast_daily_returns.mean()

# Annualize
tsla_forecast_return = (1 + tsla_mean_daily_return) ** 252 - 1

# Expected annual returns
expected_returns = np.array([tsla_forecast_return, bnd_hist_return, spy_hist_return])
cov_matrix = returns.cov().values * 252  # annualized

def portfolio_metrics(weights):
    ret = np.dot(weights, expected_returns)
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return ret, vol

# Max Sharpe Ratio
def neg_sharpe(weights, risk_free=0.02):
    ret, vol = portfolio_metrics(weights)
    return -(ret - risk_free) / vol

constraints = ({'type':'eq', 'fun': lambda w: np.sum(w) - 1})
bounds = tuple((0, 1) for _ in range(3))

max_sharpe_result = minimize(neg_sharpe, [1/3, 1/3, 1/3], bounds=bounds, constraints=constraints)
max_sharpe_weights = max_sharpe_result.x

# Min Volatility
def portfolio_vol(weights):
    return portfolio_metrics(weights)[1]

min_vol_result = minimize(portfolio_vol, [1/3, 1/3, 1/3], bounds=bounds, constraints=constraints)
min_vol_weights = min_vol_result.x

# Plot the Efficient Frontier
port_returns = []
port_vols = []
for _ in range(5000):
    w = np.random.dirichlet(np.ones(3), size=1)[0]
    ret, vol = portfolio_metrics(w)
    port_returns.append(ret)
    port_vols.append(vol)


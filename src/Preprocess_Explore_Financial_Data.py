Install & Import Required Libraries
# Install yfinance if not already installed
#!pip install yfinance

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller

 Download Historical Data
# Define tickers and date range
tickers = ["TSLA", "BND", "SPY"]
start_date = "2015-07-01"
end_date = "2025-07-31"

# Download full OHLCV data
data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=False)

# Save raw data to CSV
data.to_csv("./data/financial_data_TSLA_BND_SPY_2015-2025.csv")

print("Data shape:", data.shape)

Preview Data
# Example: View TSLA data
tsla_df = data["TSLA"].copy()
print(tsla_df.head())

# Check data ranges
for ticker in tickers:
    print(f"\n{ticker} Date Range:", data[ticker].index.min(), "to", data[ticker].index.max())

Data Cleaning & Preparation
# Check for missing values
print("Missing values:\n", data.isnull().sum())

# Fill missing values using forward fill (market closed days)
data = data.ffill()

# Ensure all columns are numeric
data = data.apply(pd.to_numeric, errors="coerce")

# Check again
print("Data types:\n", data.dtypes)

 Basic Statistics
print(data.describe().T)

Daily Returns & Volatility
# Calculate daily percentage change
returns = data.pct_change().dropna()

# Plot daily returns
plt.figure(figsize=(10,6))
returns["TSLA"].plot(label="TSLA", alpha=0.7)
returns["BND"].plot(label="BND", alpha=0.7)
returns["SPY"].plot(label="SPY", alpha=0.7)
plt.title("Daily Returns")
plt.legend()
plt.show()

# Rolling mean & std for volatility
rolling_mean = data["TSLA"].rolling(window=30).mean()
rolling_std = data["TSLA"].rolling(window=30).std()

plt.figure(figsize=(10,6))
plt.plot(data["TSLA"], label="TSLA Price")
plt.plot(rolling_mean, label="30-Day Rolling Mean")
plt.plot(rolling_std, label="30-Day Rolling Std Dev")
plt.title("TSLA Price & Volatility")
plt.legend()
plt.show()

Outlier Detection
# Identify days with returns beyond 3 std deviations
threshold = 3 * returns.std()
outliers = returns[(returns > threshold) | (returns < -threshold)]
print("Outliers:\n", outliers.dropna(how="all"))

Seasonality & Stationarity Test
# Augmented Dickey-Fuller Test for TSLA closing price
tsla_close = data["TSLA"]["Close"].dropna()
result = adfuller(tsla_close)

print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

if result[1] <= 0.05:
    print("The series is stationary.")
else:
    print("The series is non-stationary; differencing is needed.")
Risk Metrics – Value at Risk (VaR) & Sharpe Ratio
# Value at Risk (95%)
var_95 = returns.quantile(0.05)

# Sharpe Ratio (assuming risk-free rate ~0 for simplicity)
sharpe_ratios = (returns.mean() / returns.std()) * np.sqrt(252)

print("Value at Risk (95%):\n", var_95)
print("Sharpe Ratios:\n", sharpe_ratios)
# Save processed data
data.to_csv("./data/processed_financial_data_TSLA_BND_SPY_2015-2025.csv")
# Save returns data
returns.to_csv("./data/returns_TSLA_BND_SPY_2015-2025.csv")
# Save risk metrics
risk_metrics = pd.DataFrame({
    "VaR_95": var_95,
    "Sharpe_Ratio": sharpe_ratios
})
risk_metrics.to_csv("./data/risk_metrics_TSLA_BND_SPY_2015-2025.csv")


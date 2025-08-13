# TSLA Portfolio Forecast & Backtest

## Overview
This repository contains code and notebooks for time series forecasting, portfolio optimization, and strategy backtesting using TSLA, SPY, and BND.  
The project uses ARIMA and LSTM models, risk optimization, and historical backtesting.

## Environment Setup

### 1. Python Version
- Python 3.8 or higher recommended

### 2. Required Packages

Install dependencies using pip:

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install yfinance pandas numpy matplotlib scikit-learn tensorflow==2.* scipy
```

### 3. Jupyter Notebook Support

To run notebooks:

```bash
pip install notebook
```

### 4. Running the Code

- **Notebooks:** Open `.ipynb` files in Jupyter or VS Code.
- **Scripts:** Run Python scripts from the `src/` directory:

```bash
python src/Strategy_Backtesting_Code.py
```

### 5. Data

- All data is downloaded automatically via `yfinance`.
- No manual downloads required.

## Repository Structure

- `notebook/` — Jupyter notebooks for each task
- `src/` — Python scripts for backtesting and analysis
- `requirements.txt` — List of required packages

## Screenshots & Results

Include screenshots of your results and plots in your final report or submission.

## License

MIT

---

**Contact:**  
For questions, open an issue or email [your-email@example.com].

import pandas as pd
import numpy as np

def compute_sharpe_ratio(returns, risk_free_rate=0.02):
    """
    Calculate the Sharpe Ratio for a portfolio.
    
    Parameters:
    returns (pd.Series): Asset returns.
    risk_free_rate (float): Risk-free rate (default 2%).
    
    Returns:
    float: Sharpe Ratio.
    """
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / returns.std()

if __name__ == "__main__":
    df = pd.read_csv("results/portfolio_data.csv", parse_dates=["Date"], index_col="Date")
    df["Portfolio_Return"] = df.mean(axis=1)  # Simple average return
    
    sharpe_ratio = compute_sharpe_ratio(df["Portfolio_Return"])
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

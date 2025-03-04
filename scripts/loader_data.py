import yfinance as yf
import pandas as pd

def fetch_stock_data(ticker, start="2015-01-01", end="2025-01-31"):
    """
    Fetch historical stock data from Yahoo Finance.
    
    Parameters:
    ticker (str): Stock symbol (e.g., "TSLA", "BND", "SPY").
    start (str): Start date in YYYY-MM-DD format.
    end (str): End date in YYYY-MM-DD format.
    
    Returns:
    pd.DataFrame: Stock data with Date, Open, High, Low, Close, Volume, Adj Close.
    """
    stock = yf.download(ticker, start=start, end=end)
    stock.reset_index(inplace=True)
    return stock

if __name__ == "__main__":
    tsla_data = fetch_stock_data("TSLA")
    bnd_data = fetch_stock_data("BND")
    spy_data = fetch_stock_data("SPY")
    
    tsla_data.to_csv("results/TSLA.csv", index=False)
    bnd_data.to_csv("results/BND.csv", index=False)
    spy_data.to_csv("results/SPY.csv", index=False)

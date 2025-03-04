import pandas as pd
import matplotlib.pyplot as plt

def read_stock_data(filepath):
    """
    Read stock data from a CSV file.
    
    Parameters:
    filepath (str): Path to the processed CSV file.
    
    Returns:
    pd.DataFrame: Stock data with 'Date' as index.
    """
    return pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")

def plot_stock_trend(df, ticker):
    """
    Plot stock closing price trends.
    
    Parameters:
    df (pd.DataFrame): Stock data.
    ticker (str): Stock symbol (e.g., "TSLA").
    """
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["Close"], label=f"{ticker} Close Price")
    plt.title(f"{ticker} Stock Price Trend")
    plt.xlabel("Date")
    plt.ylabel("Closing Price")
    plt.legend()
    plt.grid()
    plt.show()

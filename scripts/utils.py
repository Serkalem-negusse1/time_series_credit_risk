import yfinance as yf
import pandas as pd

def fetch_data(tickers, start_date, end_date):
    """Fetch historical data for given tickers."""
    data = {ticker: yf.download(ticker, start=start_date, end=end_date) for ticker in tickers}
    df = pd.DataFrame({ticker: data[ticker]['Adj Close'] for ticker in tickers})
    df.to_csv("results/financial_data.csv")
    return df

if __name__ == "__main__":
    tickers = ["TSLA", "BND", "SPY"]
    start_date = "2015-01-01"
    end_date = "2025-01-31"
    df = fetch_data(tickers, start_date, end_date)
    print("Data saved as financial_data.csv")

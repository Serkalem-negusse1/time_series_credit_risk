import yfinance as yf
import pandas as pd
import os

def fetch_data(tickers, start_date, end_date):
    """Fetch historical data for given tickers."""
    # Download data for each ticker
    data = {ticker: yf.download(ticker, start=start_date, end=end_date) for ticker in tickers}
    
    # Ensure 'results' directory exists
    os.makedirs("results", exist_ok=True)

    # Check and print the columns available for each ticker
    for ticker in tickers:
        print(f"Data for {ticker}:", data[ticker].columns)

    # Create DataFrame using 'Adj Close' if available, else fallback to 'Close'
    df = pd.DataFrame({ticker: data[ticker]['Adj Close'] if 'Adj Close' in data[ticker].columns else data[ticker]['Close'] for ticker in tickers})
    
    # Save the DataFrame to a CSV file
    df.to_csv("results/financial_data.csv")
    return df

if __name__ == "__main__":
    tickers = ["TSLA", "BND", "SPY"]
    start_date = "2015-01-01"
    end_date = "2025-01-31"
    
    # Fetch the data and save it to CSV
    df = fetch_data(tickers, start_date, end_date)
    print("Data saved as financial_data.csv")

import yfinance as yf
import os

# Define tickers
tickers = ['TSLA', 'BND', 'SPY']

# Download historical data (from Jan 1, 2015, to Jan 31, 2025)
data = yf.download(tickers, start="2015-01-01", end="2025-01-31")

# Print the first few rows of the data
print(data.head())

# Specify the folder path where you want to save the CSV file
folder_path = "E:/data/Data11"

# Ensure the folder exists (if not, create it)
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Define the full file path
file_path = os.path.join(folder_path, "historical_financial_data.csv")

# Save the data to a CSV file
data.to_csv(file_path)

# Confirm the file has been saved
print(f"Data has been saved to: {file_path}")

# Import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from pmdarima import auto_arima
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Utility Functions - data_loader.py
def fetch_stock_data(ticker, start='2015-01-01', end='2025-01-31'):
    stock = yf.download(ticker, start=start, end=end)
    stock.reset_index(inplace=True)
    return stock

# Utility Functions - data_preprocessing.py
def preprocess_data(df):
    df = df[['Close']].dropna()
    df['Returns'] = df['Close'].pct_change().dropna()
    scaler = MinMaxScaler()
    df['Close_scaled'] = scaler.fit_transform(df[['Close']])
    return df, scaler

# Utility Functions - forecasting.py
def train_arima(df):
    model = auto_arima(df['Close'], seasonal=False, trace=True)
    return model.fit(df['Close'])

def train_sarima(df):
    model = SARIMAX(df['Close'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    return model.fit()

def train_lstm(df):
    data = df['Close_scaled'].values.reshape(-1, 1)
    X, y = [], []
    for i in range(60, len(data)):
        X.append(data[i-60:i])
        y.append(data[i])
    X, y = np.array(X), np.array(y)
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=16)
    return model

# Utility Functions - optimization.py
def compute_portfolio_metrics(preprocessed_data):
    returns = pd.DataFrame({stock: preprocessed_data[stock]['Returns'] for stock in preprocessed_data}).dropna()
    cov_matrix = returns.cov()
    avg_returns = returns.mean()
    weights = np.array([0.4, 0.3, 0.3])  # Initial allocation
    portfolio_return = np.dot(weights, avg_returns)
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    sharpe_ratio = portfolio_return / portfolio_risk
    return portfolio_return, portfolio_risk, sharpe_ratio

# Main Execution
if __name__ == "__main__":
    stocks = ['TSLA', 'BND', 'SPY']
    data = {stock: fetch_stock_data(stock) for stock in stocks}
    preprocessed_data = {stock: preprocess_data(data[stock])[0] for stock in stocks}
    scalers = {stock: preprocess_data(data[stock])[1] for stock in stocks}
    
    arima_models = {stock: train_arima(preprocessed_data[stock]) for stock in stocks}
    sarima_models = {stock: train_sarima(preprocessed_data[stock]) for stock in stocks}
    lstm_models = {stock: train_lstm(preprocessed_data[stock]) for stock in stocks}
    
    portfolio_return, portfolio_risk, sharpe_ratio = compute_portfolio_metrics(preprocessed_data)
    print(f"Portfolio Return: {portfolio_return:.2f}")
    print(f"Portfolio Risk: {portfolio_risk:.2f}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")

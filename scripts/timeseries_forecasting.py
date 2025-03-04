import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from scipy.optimize import minimize
from keras.losses import MeanSquaredError

def load_data(ticker, start, end):
    data = yf.download(ticker, start=start, end=end)
    return data[['Open', 'High', 'Low', 'Close', 'Volume']]

def preprocess_data(df):
    df = df.dropna()
    scaler = MinMaxScaler()
    df[['Open', 'High', 'Low', 'Close', 'Volume']] = scaler.fit_transform(df[['Open', 'High', 'Low', 'Close', 'Volume']])
    return df, scaler

def train_arima(df, order=(5,1,0)):
    model = ARIMA(df['Close'], order=order)
    model_fit = model.fit()
    return model_fit

def train_sarima(df, order=(1,1,1), seasonal_order=(1,1,1,12)):
    model = SARIMAX(df['Close'], order=order, seasonal_order=seasonal_order)
    model_fit = model.fit()
    return model_fit

def train_lstm(df, time_steps=10):
    X, y = [], []
    for i in range(len(df)-time_steps):
        X.append(df.iloc[i:i+time_steps, :].values)
        y.append(df.iloc[i+time_steps, 3])
    X, y = np.array(X), np.array(y)
    
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(time_steps, 5)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=10, batch_size=16)
    return model

def save_model(model, filename):
    if isinstance(model, (ARIMA, SARIMAX)):
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
    else:
        model.save(filename)

def load_model_file(filename, model_type='lstm'):
    if model_type == 'lstm':
        return load_model(filename, custom_objects={'mse': MeanSquaredError()})
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f)

def plot_forecast(actual, forecast, title):
    plt.figure(figsize=(12, 6))
    plt.plot(actual.index, actual, label='Actual', color='blue')
    plt.plot(forecast.index, forecast, label='Forecast', color='red', linestyle='dashed')
    plt.legend()
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.grid()
    plt.show()

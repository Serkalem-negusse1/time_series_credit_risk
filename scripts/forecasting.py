import pandas as pd
import statsmodels.api as sm

def train_arima(filepath, order=(5,1,0)):
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    model = sm.tsa.ARIMA(df["Close"], order=order).fit()
    return model

def train_sarima(filepath, order=(5,1,0)):
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    model1 = sm.tsa.SARIMA(df["Close"], order=order).fit()
    return model1

def train_lstm(filepath, order=(5,1,0)):
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    model2 = sm.tsa.LSTM(df["Close"], order=order).fit()
    return model2

if __name__ == "__main__":
    arima_model = train_arima("results/TSLA_preprocessed.csv")
    print(arima_model.summary())
###
    sarima_model = train_arima("results/TSLA_preprocessed.csv")
    print(sarima_model.summary())

    lstm_model = train_arima("results/TSLA_preprocessed.csv")
    print(lstm_model.summary())

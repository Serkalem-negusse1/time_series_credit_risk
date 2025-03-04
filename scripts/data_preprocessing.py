import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(filepath):
    """
    Load, clean, and preprocess stock data.
    
    Parameters:
    filepath (str): Path to the raw CSV file.
    
    Returns:
    pd.DataFrame: Cleaned data.
    """
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df.dropna(inplace=True)  # Remove missing values
    df.set_index("Date", inplace=True)

    # Scaling Close price for ML models
    scaler = MinMaxScaler()
    df["Close_scaled"] = scaler.fit_transform(df[["Close"]])
    
    return df

if __name__ == "__main__":
    tsla_clean = preprocess_data("results/TSLA.csv")
    tsla_clean.to_csv("results/TSLA_preprocessed.csv")

    bnd_clean = preprocess_data("results/BND.csv")
    bnd_clean.to_csv("results/BND_preprocessed.csv")

    spy_clean = preprocess_data("results/SPY.csv")
    spy_clean.to_csv("results/spy_preprocessed.csv")
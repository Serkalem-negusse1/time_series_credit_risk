import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_data(file_path):
    """Load and clean the preprocessed dataset."""
    df = pd.read_csv(file_path, parse_dates=True, index_col=0)
    
    # Drop the 'Ticker' row which repeats ticker names for columns
    df = df.drop(index='Ticker', errors='ignore')  # Ignore errors if 'Ticker' row is not found
    
    # Rename columns to make them clear (e.g., TSLA_Price, BND_Adj Close, etc.)
    df.columns = [f"{col}_{ticker}" for col, ticker in zip(df.columns, df.iloc[0, :])]
    
    # Drop the first row after renaming columns (which contains the tickers)
    df = df.drop(index=df.index[0])  # Drop the first row by its position
    
    print(df.columns)  # Print the columns to inspect the new names
    return df

def split_data(df):
    """Split data into training and testing sets."""
    X = df.drop(columns=['Adj Close_TSLA', 'Close.2_TSLA', 'High.2_TSLA', 'Low.2_TSLA', 'Open.2_TSLA', 'Volume.2_TSLA'])  # Use other assets to predict TSLA's price
    y = df['Close.2_TSLA']  # Target is TSLA's close price (or you can use 'Adj Close_TSLA' or another column)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_model(X_train, y_train):
    """Train a Random Forest model."""
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    return y_pred

if __name__ == "__main__":
    # Load and clean data
    df = load_data("results/preprocessed_data.csv")
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = split_data(df)
    
    # Train the model
    model = train_model(X_train, y_train)
    
    # Evaluate the model
    evaluate_model(model, X_test, y_test)

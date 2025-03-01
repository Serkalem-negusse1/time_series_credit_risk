import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    """Load the dataset."""
    return pd.read_csv(file_path, parse_dates=True, index_col=0)

def handle_missing_values(df):
    """Fill missing values using forward-fill and mean imputation for numeric columns, 
    and most frequent imputation for categorical columns."""
    
    # Forward fill for missing values in all columns
    df.ffill(inplace=True)  # Forward fill
    
    # Handle numeric columns
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if numeric_columns.size > 0:
        imputer_numeric = SimpleImputer(strategy="mean")
        df[numeric_columns] = imputer_numeric.fit_transform(df[numeric_columns])
    else:
        print("No numeric columns found for imputation.")
    
    # Handle categorical columns (non-numeric data)
    categorical_columns = df.select_dtypes(include=['object']).columns
    if categorical_columns.size > 0:
        imputer_categorical = SimpleImputer(strategy="most_frequent")
        df[categorical_columns] = imputer_categorical.fit_transform(df[categorical_columns])
    else:
        print("No categorical columns found for imputation.")
    
    return df

def normalize_data(df):
    """Normalize data using MinMaxScaler for numeric columns only."""
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    if numeric_columns.size > 0:
        scaler = MinMaxScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    else:
        print("No numeric columns found for normalization.")
    return df

if __name__ == "__main__":
    try:
        # Load the data
        df = load_data("E:/data/Data11/historical_financial_data.csv")
        
        # Handle missing values
        df = handle_missing_values(df)
        
        # Normalize data
        df = normalize_data(df)
        
        # Save the preprocessed data
        df.to_csv("E:/Git_repo/time_series_credit_risk/results/preprocessed_data.csv", index=True)
        print("Preprocessing complete. Data saved as preprocessed_data.csv")
    
    except Exception as e:
        print(f"An error occurred during preprocessing: {e}")

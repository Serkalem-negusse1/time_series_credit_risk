import pandas as pd

# Load the dataset, skipping the first row (Ticker row)
df = pd.read_csv("E:/data/Data11/preprocessed_financial_data.csv", header=[0], index_col=0)  # Ensure Date is used as index

# Rename columns to remove duplicate names (if needed)
df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True)  # Remove spaces

# Check if the columns are cleaned
print(df.head())  # Ensure proper format

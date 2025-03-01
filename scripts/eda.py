import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
#from tabulate import tabulate
from prettytable import PrettyTable
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path, parse_dates=True, index_col=0)

def clean_data(df):
    """Clean and restructure the dataset."""
    # Flatten multi-level columns (if any)
    df.columns = [' '.join(col).strip() for col in df.columns.values]
    return df

def summary_statistics(df):
    """Print summary statistics of the dataset in a properly formatted table."""
    df.columns = df.columns.str.strip().str.replace(r'\s+', ' ', regex=True)
    stats = df.describe()
    table = PrettyTable()
    table.field_names = ["Statistic"] + list(stats.columns) 

    for index, row in stats.iterrows():
        table.add_row([index] + list(row))

    print(table)
    return stats  


def missing_values(df):
    """Check for missing values."""
    print("Missing Values:\n", df.isnull().sum())

def visualize_data(df):
    """Plot historical trends for all assets."""
    # Ensure the index is in datetime format (if it's not already)
    df.index = pd.to_datetime(df.index, errors='coerce')

    plt.figure(figsize=(12, 6))
    
    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
        plt.plot(df.index, df[column], label=str(column))
    
    plt.legend()
    plt.title("Stock Prices Over Time")
    plt.xlabel("Date")
    plt.ylabel("Adjusted Close Price")
    plt.grid()
    plt.show()

def correlation_heatmap(df):
    """Visualize Correlation Between Features."""
    plt.figure(figsize=(10, 6))
    corr_matrix = df.corr()
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

def missing_data_visualization(df):
    """Visualize missing data in the dataset."""
    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
    plt.title("Missing Data Visualization")
    plt.show()

def plot_histograms(df):
    """Plot histograms for numerical features."""
    df_numeric = df.select_dtypes(include=['float64', 'int64'])
    df_numeric.hist(bins=30, figsize=(15, 10))
    plt.suptitle("Histograms of Numerical Features", size=16)
    plt.show()

def plot_boxplots(df):
    """Plot boxplots to detect outliers."""
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=df.select_dtypes(include=['float64', 'int64']), orient="h", palette="Set2")
    plt.title("Boxplots to Detect Outliers")
    plt.show()

def seasonal_decomposition(df, column, period=30):
 import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def seasonal_decomposition(df, column, period=30):
    df = df.copy()  

    try:
        df.index = pd.to_datetime(df.index, errors='coerce', infer_datetime_format=True)
    except Exception as e:
        raise ValueError(f"Error parsing datetime index: {e}")

    df = df.dropna(subset=[column])
    
    df[column] = pd.to_numeric(df[column], errors='coerce')

    if len(df) < period:
        raise ValueError(f"Not enough data points ({len(df)}) for the specified period ({period})")

    decomposition = sm.tsa.seasonal_decompose(df[column], model='additive', period=period)
    decomposition.plot()
    plt.show()


def autocorrelation_plots(df, column):
    """Plot autocorrelation and partial autocorrelation."""
    df[column] = pd.to_numeric(df[column], errors='coerce')
    
    plt.figure(figsize=(12, 6))
    plot_acf(df[column], lags=50)
    plt.show()

    plt.figure(figsize=(12, 6))
    plot_pacf(df[column], lags=50)
    plt.show()

def save_cleaned_data(df, file_path):
    """Save the cleaned dataset to a new CSV file."""
    df.to_csv(file_path)
    print(f"Data saved to {file_path}")

df = pd.read_csv("E:/data/Data11/preprocessed_financial_data.csv", header=[0], index_col=0)  # Ensure Date is used as index
df.columns = df.columns.str.strip().str.replace(r'\s+', '_', regex=True)  # Remove spaces


# Main script for running everything
if __name__ == "__main__":
    # Define the file path
    file_path = "E:/data/Data11/historical_financial_data.csv"
    
    # Load the raw data
    df = load_data(file_path)
    
    # Clean the data
    df_cleaned = clean_data(df)
    
    # Save the cleaned data with a different name
    save_cleaned_data(df_cleaned, "E:/data/Data11/preprocessed_financial_data.csv")
    
    # Print summary statistics and check for missing values
    #summary_statistics(df_cleaned)
    #missing_values(df_cleaned)
    
    # Visualize the data
    #visualize_data(df_cleaned)
    
    # Additional visualizations
    #correlation_heatmap(df_cleaned)
    #autocorrelation_plots(df_cleaned)
    #seasonal_decomposition(df_cleaned)
    #missing_data_visualization(df_cleaned)
    #plot_histograms(df_cleaned)
    #plot_boxplots(df_cleaned)

import os
import pandas as pd
from datetime import datetime, timedelta

def load_data(file_path):
    """
    Reads the Excel file and returns the loaded dataframe
    """
    try:
        df = pd.read_excel(file_path)
        print(f"Data loaded successfully from {file_path}.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

def clean_data(df):
    """
    Cleans the dataset by removing missing values and anomalies and returns a cleaned dataframe.
    """
    initial_shape = df.shape
    # Change Customer ID columns to ColumnID
    df.rename(columns= {'Customer ID':'CustomerID'}, inplace=True)
    # Remove rows with missing CustomerID
    df_clean = df.dropna(subset=['CustomerID'])
    # Remove rows with non-positive Quantity
    df_clean = df_clean[df_clean['Quantity'] > 0]
    # Remove rows with negative or zero Price
    df_clean = df_clean[df_clean['Price'] > 0]
    final_shape = df_clean.shape
    print(f"Data cleaned: {initial_shape[0] - final_shape[0]} rows removed.")
    return df_clean

def calculate_rfm(df):
    """
    Calculates RFM metrics for each customer.
    """
    # Define analysis date as one day after the last InvoiceDate
    analysis_date = df['InvoiceDate'].max() + timedelta(days=1)
    print(f"Analysis Date set to: {analysis_date}")

    # Calculate RFM metrics
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (analysis_date - x.max()).days,
        'Invoice': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()

    # Rename columns
    rfm.rename(columns={
        'InvoiceDate': 'Recency',
        'Invoice': 'Frequency',
        'TotalPrice': 'Monetary'
    }, inplace=True)

    print("RFM metrics calculated successfully.")
    return rfm

def add_total_price(df):
    """
    Adds a TotalPrice column to the DataFrame.
    """
    df['TotalPrice'] = df['Quantity'] * df['Price']
    print("TotalPrice column added.")
    return df

def save_rfm(rfm, output_path):
    """
    Saves the RFM metrics to a CSV file.
    """
    try:
        rfm.to_csv(output_path, index=False)
        print(f"RFM metrics saved to {output_path}.")
    except Exception as e:
        print(f"Error saving RFM metrics: {e}")
        raise

def main():
    # Define file paths
    raw_data_path = os.path.join('..', 'raw_data', 'online_retail_II.xlsx')
    rfm_output_path = os.path.join('..', 'raw_data', 'rfm_with_clusters.csv')

    # Load data
    df = load_data(raw_data_path)

    # Add TotalPrice
    df = add_total_price(df)

    # Clean data
    df_clean = clean_data(df)

    # Calculate RFM metrics
    rfm = calculate_rfm(df_clean)

    # Save RFM metrics
    save_rfm(rfm, rfm_output_path)

if __name__ == "__main__":
    main()

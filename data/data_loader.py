import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Paths for data storage
CSV_FILE = 'stocks_data.csv'
PARQUET_FILE = 'stocks_data.parquet'

# Fetch tickers from S&P 600 and S&P 400
sp600_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies')[0]['Symbol'].tolist()
sp400_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies')[0]['Symbol'].tolist()
sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
tickers = sp500_tickers + sp600_tickers + sp400_tickers

# Function to download stock data
def download_stock_data(tickers, start_date, end_date):
    all_data = []
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            data.reset_index(inplace=True)
            data['Stock_Ticker'] = ticker
            data['Stock_Name'] = ticker  # Adjust if actual names are available
            data = data[['Date', 'Close', 'Stock_Ticker', 'Stock_Name']]
            data.rename(columns={'Close': 'Price'}, inplace=True)
            data['Date'] = data['Date'].dt.strftime('%d/%m/%Y')
            all_data.append(data)
            print(f'Downloaded data for {ticker}')
        except Exception as e:
            print(f'Failed to download data for {ticker}: {e}')
    
    return pd.concat(all_data, ignore_index=True)

# Function to update data files
def update_data_files():
    if os.path.exists(CSV_FILE) and os.path.exists(PARQUET_FILE):
        # Load existing data to get the last date
        existing_data = pd.read_csv(CSV_FILE)
        existing_data['Date'] = pd.to_datetime(existing_data['Date'], dayfirst=True)
        last_date = existing_data['Date'].max()
        start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
        print(f'Updating data from {start_date}')
    else:
        start_date = '2000-01-01'
        print(f'Downloading initial data from {start_date}')

    end_date = datetime.today().strftime('%Y-%m-%d')

    if pd.to_datetime(start_date) > pd.to_datetime(end_date):
        print('Data is already up-to-date.')
        return

    # Download new data
    new_data = download_stock_data(tickers, start_date, end_date)

    if new_data.empty:
        print('No new data to append.')
        return

    if os.path.exists(CSV_FILE) and os.path.exists(PARQUET_FILE):
        # Append new data
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        print('Appending new data.')
    else:
        updated_data = new_data
        print('Creating new data files.')

    # Save data to CSV and Parquet
    updated_data.to_csv(CSV_FILE, index=False)
    updated_data.to_parquet(PARQUET_FILE, engine='pyarrow', index=False)
    print(f'Data successfully saved to {CSV_FILE} and {PARQUET_FILE}.')


# Execute the update
if __name__ == '__main__':
    update_data_files()

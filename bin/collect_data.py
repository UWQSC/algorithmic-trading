"""
Stock data loader module that fetches stock tickers from S&P indices
and downloads historical price data.
"""

import os
import time

from datetime import datetime, timedelta
from typing import Optional, List

import pandas as pd
import yfinance as yf

from uwqsc_algorithmic_trading.src.common.config import PATHS


def get_combined_sp_tickers() -> List[str]:
    """
    Fetch stock tickers from S&P indices and return as a combined list.

    Returns:
        list: List of unique stock ticker symbols from S&P 500, 400 and 600 indices
    """
    print("Fetching ticker lists...")
    try:
        # Fetch tickers from Wikipedia
        sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        sp400_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
        sp600_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'

        sp500_tickers = pd.read_html(sp500_url)[0]['Symbol'].tolist()
        sp400_tickers = pd.read_html(sp400_url)[0]['Symbol'].tolist()
        sp600_tickers = pd.read_html(sp600_url)[0]['Symbol'].tolist()

        # Clean tickers: remove any that contain dots
        all_tickers = [
            ticker for ticker in sp400_tickers + sp600_tickers + sp500_tickers
            if '.' not in ticker and len(ticker) > 0
        ]

        # Remove duplicates
        all_tickers = list(set(all_tickers))
        print(f"Found {len(all_tickers)} unique tickers")

        return all_tickers
    except (ValueError, IndexError) as error:
        print(f"Error fetching tickers: {error}")
        return []


def get_stock_data(ticker_file='sp_ticker.csv',
                   output_csv='combined_stock_data.csv',
                   output_parquet='combined_stock_data.parquet') -> Optional[pd.DataFrame]:
    """
    Retrieves historical stock data for tickers listed in a CSV file
    and combines them into a single DataFrame.

    Args:
        ticker_file (str): Path to CSV file containing stock tickers (must have column 'Ticker')
        output_csv (str): Path to save combined data in CSV format
        output_parquet (str): Path to save combined data in Parquet format

    Returns:
        pandas.DataFrame: Combined stock data with columns: Date, Stock_Name, Stock_Ticker, Price
    """
    # Initialize variables
    combined_data = pd.DataFrame()
    start_date = "2000-01-01"  # Start from January 1st, 2000
    end_date = datetime.now().strftime('%Y-%m-%d')

    # Read the ticker list
    try:
        print("Reading ticker list from", ticker_file)
        ticker_df = pd.read_csv(ticker_file)

        if 'Ticker' not in ticker_df.columns:
            print("Error: CSV file must contain a column named 'Ticker'")
            return None

        tickers = ticker_df['Tickers'].tolist()
        print("Found", len(tickers), "tickers in the input file")
    except (FileNotFoundError, pd.errors.EmptyDataError) as error:
        print("Error reading ticker file:", str(error))
        return None

    # Check if output files exist and get the latest date
    if os.path.exists(output_csv):
        try:
            existing_data = pd.read_csv(output_csv)
            if not existing_data.empty:
                # Convert existing dates to datetime for comparison
                existing_data['Date'] = pd.to_datetime(existing_data['Date'], format='%d/%m/%Y')
                latest_date = existing_data['Date'].max()
                # Set start date to day after the latest date
                start_date = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
                print("Found existing data up to", latest_date.strftime('%d/%m/%Y'))
                print("Will update data from", start_date)

                # Convert back to string format for storage
                existing_data['Date'] = existing_data['Date'].dt.strftime('%d/%m/%Y')
                combined_data = existing_data
        except (pd.errors.ParserError, KeyError) as error:
            print("Error reading existing CSV file:", str(error))
            print("Will create new output files")

    # Process each ticker
    for i, ticker in enumerate(tickers):
        try:
            print("Processing ticker", ticker, "(", i+1, "of", len(tickers), ")")

            # Get stock data
            stock = yf.Ticker(ticker)
            hist_data = stock.history(start=start_date, end=end_date)

            if hist_data.empty:
                print("No data available for", ticker, "in the specified date range")
                continue

            # Get stock name
            try:
                stock_name = stock.info.get('shortName', ticker)
            except (KeyError, TypeError):
                stock_name = ticker

            # Process the data
            ticker_data = pd.DataFrame({
                'Date': hist_data.index.strftime('%d/%m/%Y'),
                'Stock_Name': stock_name,
                'Stock_Ticker': ticker,
                'Price': hist_data['Close']
            })

            # Append to combined data
            combined_data = pd.concat([combined_data, ticker_data], ignore_index=True)

            # Avoid rate limiting
            time.sleep(0.5)

        except Exception as error:  # pylint: disable=broad-except
            print("Error processing ticker", ticker, ":", str(error))

    # Remove duplicates if any
    combined_data.drop_duplicates(subset=['Date', 'Stock_Ticker'], inplace=True)
    print("Combined data contains", len(combined_data), "records")

    # Save to output files
    try:
        if not combined_data.empty:
            # Save to CSV
            combined_data.to_csv(output_csv, index=False)
            print("Data saved to", output_csv)

            # Save to Parquet
            combined_data.to_parquet(output_parquet, index=False)
            print("Data saved to", output_parquet)
        else:
            print("No data to save")
    except (IOError, ValueError) as error:
        print("Error saving output files:", str(error))

    return combined_data


if __name__ == '__main__':
    # Get tickers from S&P indices
    SP_TICKERS = get_combined_sp_tickers()
    DATA_DIR = PATHS.DATA_DIR

    # Export tickers to CSV in the data directory
    TICKERS_DF = pd.DataFrame(SP_TICKERS, columns=['Tickers'])
    CSV_PATH = os.path.join(DATA_DIR, 'sp_tickers.csv')
    TICKERS_DF.to_csv(CSV_PATH, index=False)

    # Get stock data for all tickers
    get_stock_data(
        ticker_file=CSV_PATH,
        output_csv=os.path.join(DATA_DIR, 'combined_stock_data.csv'),
        output_parquet=os.path.join(DATA_DIR, 'combined_stock_data.parquet')
    )

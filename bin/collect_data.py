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

## CONSTANTS
TAG = "[COLLECT DATA]"

SYMBOL_COLUMN = "Symbol"
TICKER_COLUMN = "Tickers"
DATE_COLUMN = "Date"
STOCK_NAME_COLUMN = "Stock_Name"
STOCK_TICKER_COLUMN = "Stock_Ticker"
PRICE_COLUMN = "Price"
CLOSE_COLUMN = "Close"

TICKERS_FILE = "sp_tickers.csv"
STOCK_DATA = "combined_stock_data.csv"
STOCK_DATA_PARQUET = 'combined_stock_data.parquet'

DATA_DIR = PATHS.DATA_DIR
CSV_PATH = os.path.join(DATA_DIR, TICKERS_FILE)
OUTPUT_CSV = os.path.join(DATA_DIR, STOCK_DATA)
OUTPUT_PARQUET = os.path.join(DATA_DIR, STOCK_DATA_PARQUET)


def get_combined_sp_tickers() -> List[str]:
    """
    Fetch stock tickers from S&P indices and return as a combined list.

    Returns:
        list: List of unique stock ticker symbols from S&P 500, 400 and 600 indices
    """
    print(f"{TAG} Fetching ticker lists...")
    try:
        # Fetch tickers from Wikipedia
        sp500_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        sp400_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies'
        sp600_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_600_companies'

        sp500_tickers = pd.read_html(sp500_url)[0][SYMBOL_COLUMN].tolist()
        sp400_tickers = pd.read_html(sp400_url)[0][SYMBOL_COLUMN].tolist()
        sp600_tickers = pd.read_html(sp600_url)[0][SYMBOL_COLUMN].tolist()

        # Clean tickers: remove any that contain dots
        all_tickers = [
            ticker for ticker in sp400_tickers + sp600_tickers + sp500_tickers
            if '.' not in ticker and len(ticker) > 0
        ]

        # Remove duplicates
        all_tickers = list(set(all_tickers))
        print(f"{TAG} Found {len(all_tickers)} unique tickers")

        return all_tickers
    except (ValueError, IndexError) as error:
        print(f"{TAG} Error fetching tickers: {error}")
        return []


def get_stock_data() -> Optional[pd.DataFrame]:
    """
    Retrieves historical stock data for tickers listed in a CSV file
    and combines them into a single DataFrame.

    Returns:
        pandas.DataFrame: Combined stock data with columns: Date, Stock_Name, Stock_Ticker, Price
    """
    # Initialize variables
    combined_data = pd.DataFrame()
    start_date = "2000-01-01"  # Start from January 1st, 2000
    end_date = datetime.now().strftime('%Y-%m-%d')

    # Read the ticker list
    try:
        print(f"{TAG} Reading ticker list from {TICKERS_FILE}")
        ticker_df = pd.read_csv(CSV_PATH)

        if TICKER_COLUMN not in ticker_df.columns:
            print(f"{TAG} Error: CSV file must contain a column named {TICKER_COLUMN}")
            return None

        tickers = ticker_df[TICKER_COLUMN].tolist()
        print(f"{TAG} Found {len(tickers)} tickers in the input file")
    except (FileNotFoundError, pd.errors.EmptyDataError) as error:
        print(f"{TAG} Error reading ticker file: {str(error)}")
        return None

    # Check if output files exist and get the latest date
    if os.path.exists(OUTPUT_CSV):
        try:
            existing_data = pd.read_csv(OUTPUT_CSV)
            if not existing_data.empty:
                # Convert existing dates to datetime for comparison
                existing_data[DATE_COLUMN] = pd.to_datetime(
                    existing_data[DATE_COLUMN],
                    format='%d/%m/%Y'
                )
                latest_date = existing_data[DATE_COLUMN].max()
                # Set start date to day after the latest date
                start_date = (latest_date + timedelta(days=1)).strftime('%Y-%m-%d')
                print(f"{TAG} Found existing data up to  {latest_date.strftime('%d/%m/%Y')}")
                print(f"{TAG} Will update data from {start_date}")

                # Convert back to string format for storage
                existing_data[DATE_COLUMN] = existing_data[DATE_COLUMN].dt.strftime('%d/%m/%Y')
                combined_data = existing_data
        except (pd.errors.ParserError, KeyError) as error:
            print(f"{TAG} Error reading existing CSV file: {str(error)}")
            print(f"{TAG} Will create new output files")

    # Process each ticker
    for i, ticker in enumerate(tickers):
        try:
            print(f"{TAG} Processing ticker {ticker} ({i + 1} of {len(tickers)})")

            # Get stock data
            stock = yf.Ticker(ticker)
            hist_data = stock.history(start=start_date, end=end_date)

            if hist_data.empty:
                print(f"{TAG} No data available for {ticker} in the specified date range")
                continue

            # Get stock name
            try:
                stock_name = stock.info.get('shortName', ticker)
            except (KeyError, TypeError):
                stock_name = ticker

            # Process the data
            ticker_data = pd.DataFrame({
                DATE_COLUMN: hist_data.index.strftime('%d/%m/%Y'),
                STOCK_NAME_COLUMN: stock_name,
                STOCK_TICKER_COLUMN: ticker,
                PRICE_COLUMN: hist_data[CLOSE_COLUMN]
            })

            # Append to combined data
            combined_data = pd.concat([combined_data, ticker_data], ignore_index=True)

            # Avoid rate limiting
            time.sleep(0.5)

        except Exception as error:  # pylint: disable=broad-except
            print(f"{TAG} Error processing ticker {ticker} : {str(error)}")

    # Remove duplicates if any
    combined_data.drop_duplicates(subset=[DATE_COLUMN, STOCK_TICKER_COLUMN], inplace=True)
    print(f"{TAG} Combined data contains {len(combined_data)} records")

    # Save to output files
    try:
        if not combined_data.empty:
            # Save to CSV
            combined_data.to_csv(OUTPUT_CSV, index=False)
            print(f"{TAG} Data saved to {OUTPUT_CSV}")

            # Save to Parquet
            combined_data.to_parquet(OUTPUT_PARQUET, index=False)
            print(f"{TAG} Data saved to {OUTPUT_PARQUET}")
        else:
            print(f"{TAG} No data to save")
    except (IOError, ValueError) as error:
        print(f"{TAG} Error saving output files: {str(error)}")

    return combined_data


if __name__ == '__main__':
    # Get tickers from S&P indices
    SP_TICKERS = get_combined_sp_tickers()

    # Export tickers to CSV in the data directory
    TICKERS_DF = pd.DataFrame(SP_TICKERS, columns=[TICKER_COLUMN])
    TICKERS_DF.to_csv(CSV_PATH, index=False)

    # Get stock data for all tickers
    get_stock_data()

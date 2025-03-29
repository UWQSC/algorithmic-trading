import pandas as pd
import os
import yfinance as yf
from datetime import datetime, timedelta
import plotly
import plotly.graph_objects as go
import plotly.express as px
import sys
import time

def get_combined_sp_tickers():
    """Fetch stock tickers from S&P indices and return as a combined list"""
    print("Fetching ticker lists...")
    try:
        # Fetch tickers from Wikipedia
        sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()
        sp400_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_400_companies')[0]['Symbol'].tolist()
        sp600_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_600_companies')[0]['Symbol'].tolist()
        
        # Clean tickers: remove any that contain dots
        all_tickers = [ticker for ticker in sp400_tickers + sp600_tickers + sp500_tickers
                       if '.' not in ticker and len(ticker) > 0]
        
        # Remove duplicates
        all_tickers = list(set(all_tickers))
        print(f"Found {len(all_tickers)} unique tickers")
        
        return all_tickers
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return []
    
def get_stock_data(ticker_file='sp_ticker.csv', output_csv='combined_stock_data.csv', output_parquet='combined_stock_data.parquet'):
    """
    Retrieves historical stock data for tickers listed in a CSV file and combines them into a single DataFrame.
    
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
    latest_date = None
    
    # Read ticker list
    try:
        print("Reading ticker list from", ticker_file)
        ticker_df = pd.read_csv(ticker_file)
        
        if 'Ticker' not in ticker_df.columns:
            print("Error: CSV file must contain a column named 'Ticker'")
            return None
            
        tickers = ticker_df['Ticker'].tolist()
        print("Found", len(tickers), "tickers in the input file")
    except Exception as e:
        print("Error reading ticker file:", str(e))
        return None
    
    # Check if output files exist and get latest date
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
        except Exception as e:
            print("Error reading existing CSV file:", str(e))
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
            except:
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
            
        except Exception as e:
            print("Error processing ticker", ticker, ":", str(e))
    
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
    except Exception as e:
        print("Error saving output files:", str(e))
    
    return combined_data


if __name__ == '__main__':
    tickers = get_combined_sp_tickers()
    # Export tickers to CSV using the current working directory (Jupyter notebooks don't define __file__)
    data_dir = os.getcwd()
    tickers_df = pd.DataFrame(tickers, columns=['Ticker'])
    csv_path = os.path.join(data_dir, 'sp_tickers.csv')
    tickers_df.to_csv(csv_path, index=False)
    get_stock_data(ticker_file=csv_path, output_csv='combined_stock_data.csv', output_parquet='combined_stock_data.parquet')
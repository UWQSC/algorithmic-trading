import pandas as pd
import os
import yfinance as yf
from datetime import datetime, timedelta

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

def get_stock_data(ticker, data_dir=None):
    """
    Get historical stock price data for a given ticker and save to a CSV file.
    
    Args:
        ticker (str): Stock ticker symbol
        data_dir (str, optional): Directory to save data. Defaults to script directory.
        
    Returns:
        str: Path to the saved CSV file
    """
    if data_dir is None:
        data_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create the data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, f"{ticker}_price_data.csv")
    
    # Default start date is Jan 1, 2000
    start_date = "2000-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Check if file exists and get the last date
    if os.path.exists(file_path):
        try:
            existing_data = pd.read_csv(file_path)
            if not existing_data.empty:
                # Get the last date and add one day to avoid duplication
                last_date = pd.to_datetime(existing_data['Date'], format='%d/%m/%Y').max()
                start_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")
                
                # If the last date is today or in the future, no need to update
                if start_date > end_date:
                    print(f"Data for {ticker} is already up to date.")
                    return file_path
                
                print(f"Updating {ticker} data from {start_date} to {end_date}")
            else:
                print(f"Existing file for {ticker} is empty. Fetching all data.")
        except Exception as e:
            print(f"Error reading existing file for {ticker}: {e}")
            print(f"Fetching all data for {ticker}")
    else:
        print(f"Fetching historical data for {ticker} from {start_date} to {end_date}")
    
    # Fetch data from Yahoo Finance
    try:
        stock = yf.Ticker(ticker)
        hist_data = stock.history(start=start_date, end=end_date)
        
        if hist_data.empty:
            print(f"No data available for {ticker} in the specified date range.")
            return file_path
        
        # Get the stock name
        try:
            stock_name = stock.info.get('shortName', ticker)
        except:
            stock_name = ticker
        
        # Process the data
        df = pd.DataFrame({
            'Date': hist_data.index.strftime('%d/%m/%Y'),
            'Stock_Ticker': ticker,
            'Stock_Name': stock_name,
            'Price': hist_data['Close']
        })
        
        # Save or append to file
        if os.path.exists(file_path) and not existing_data.empty:
            df.to_csv(file_path, mode='a', header=False, index=False)
            print(f"Appended {len(df)} new rows to {file_path}")
        else:
            df.to_csv(file_path, index=False)
            print(f"Saved {len(df)} rows to {file_path}")
        
        return file_path
    
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

# Example usage
if __name__ == '__main__':
    tickers = get_combined_sp_tickers()
    # Export tickers to CSV
    data_dir = os.path.dirname(os.path.abspath(__file__))
    tickers_df = pd.DataFrame(tickers, columns=['Ticker'])
    csv_path = os.path.join(data_dir, 'sp_tickers.csv')
    tickers_df.to_csv(csv_path, index=False)
    print(f"Saved {len(tickers)} tickers to {csv_path}")
    print(f"Total tickers: {len(tickers)}")
    print(f"First 10 tickers: {tickers[:10]}")
    get_stock_data("AAPL")


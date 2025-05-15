"""
Implementation of the Hidden Markov Model (HMM) algorithm.
"""

from typing import Dict, List, Any

import numpy as np
from sklearn.preprocessing import StandardScaler
from hmmlearn.hmm import GaussianHMM

import pandas as pd
from pandas import DataFrame

from uwqsc_algorithmic_trading.interfaces.algorithms.algorithm_interface import IAlgorithm, StockPosition
from uwqsc_algorithmic_trading.src.preprocessing.hmm_preprocessor_impl import HMMPreProcessorImpl


class HiddenMarkovModelImpl(IAlgorithm):
    """
    Working logic for Hidden Markov Model (HMM) algorithm.
    """

    def __init__(self,
                 tickers: List[str],
                 parameters: Dict[str, Any] = None):
        '''
        Initializes the Hidden Markov Model algorithm.
        
        tickers: List of stock tickers to trade (e.g., ["AAPL", "GOOG"]).
        parameters: Dictionary of algorithm parameters (e.g., position size, HMM settings).
        '''

        name = "Hidden Markov Model" # Name of the algorithm
        data_processor = HMMPreProcessorImpl()
        
        # Dictionary to store the current positions (LONG, SHORT, HOLD) for each ticker
        # Single underscore allows test access while keeping it protected
        self._positions: Dict[str, StockPosition] = {}  # single underscore for test access
        
        # Call the parent class constructor to initialize shared functionality
        super().__init__(name, tickers, data_processor, parameters)

    def __train_hmm(self, features_scaled: np.ndarray) -> GaussianHMM:
        """
        Train a Hidden Markov Model (HMM) using the given data.

        :features_scaled: The input data (e.g., stock features) that has been standardized.
        :return: A trained HMM model.
        """
        # Create an instance of the Gaussian Hidden Markov Model (HMM)
        # n_components=3: Specifies that the HMM will have 3 hidden states (e.g., bullish, bearish, neutral)
        # covariance_type="diag": Assumes that the features are independent, so the covariance matrix is diagonal
        # n_iter=100: Sets the maximum number of iterations for the training process
        hmm = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)

        # Train the HMM on the input data
        hmm.fit(features_scaled)

        # Return the trained HMM model
        return hmm

    def generate_signals(self, current_data: DataFrame):
        """
        Generate trading signals (e.g., buy, sell, hold) based on the HMM.

        :param current_data: A DataFrame containing the latest stock price data.
        """
        # List to store features (e.g., log returns) for all stocks
        features = []

        # Loop through each stock ticker
        for ticker in self.tickers:
            # Get the column name for the stock's price
            price_col = f"{ticker}_price"

            # Check if the price column exists in the data
            if price_col in current_data.columns:
                # Get the stock prices
                prices = current_data[price_col]

                # Calculate log returns (a way to measure price changes)
                log_returns = prices.pct_change().apply(lambda x: np.log(1 + x))

                # Add the log returns to the features list (replace missing values with 0)
                features.append(log_returns.fillna(0).values)
            else:
                # Print a warning if the price column is missing
                print(f"Warning: Column '{price_col}' not found in current_data.")

        # If no features were found, raise an error
        if not features:
            raise ValueError("No valid features found. Make sure the data has the required price columns.")

        # Combine all features into a single 2D array (rows = time steps, columns = stocks)
        features = np.column_stack(features)

        # Standardize the features (make them have a mean of 0 and a standard deviation of 1)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # Train the HMM using the standardized features
        hmm = self.__train_hmm(features_scaled)

        # Use the HMM to predict the hidden market states (e.g., bullish, bearish, neutral)
        market_states = hmm.predict(features_scaled)

        # Assign trading signals based on the predicted market state
        for ticker in self.tickers:
            # Get the most recent market state
            state = market_states[-1]

            # Map the state to a trading signal
            if state == 0:
                self._positions[ticker] = StockPosition.LONG  # Buy the stock
            elif state == 1:
                self._positions[ticker] = StockPosition.SHORT
            else:
                self._positions[ticker] = StockPosition.HOLD

    def calculate_position_size(self, ticker: str, price: float, portfolio_value: float) -> float:
        """
        Calculate how many shares to buy or sell for a given stock based on the current signal.

        :param ticker: The stock ticker (e.g., "AAPL").
        :param price: The current price of the stock.
        :param portfolio_value: The total value of the portfolio.
        :return: The number of shares to trade (positive for buy, negative for sell, 0 for hold).
        """
        # Calculate the base amount of money to use for this trade.
        # This is a percentage of the total portfolio value, defined by the "position_size" parameter.
        # If "position_size" is not provided, it defaults to 10% (0.1).
        base_position_size: float = self.parameters.get("position_size", 0.1) * portfolio_value

        # Get the current signal for the stock (LONG, SHORT, or HOLD).
        signal = self._positions.get(ticker)

        # If the signal is LONG (buy), calculate how many shares to buy.
        if signal == StockPosition.LONG:
            return base_position_size / price

        # If the signal is SHORT (sell), calculate how many shares to sell (negative value).
        elif signal == StockPosition.SHORT:
            return -1 * (base_position_size / price)

        # If the signal is HOLD (do nothing), return 0 (no shares to trade).
        else:
            return 0.0

    @DeprecationWarning
    def execute_trades(self, capital: float) -> DataFrame:
        """
       Execute trades based on signals and manage portfolio.
       This implementation follows the same pattern as SimpleMovingAverageImpl.execute_trades
       but uses HMM-specific signals and position calculations.


       :param capital: Value of cash allocated to the algorithm
       :returns: DataFrame with portfolio performance tracking capital changes
       """
        # Initialize portfolio DataFrame with same index as data
        portfolio = DataFrame(index=self.__data__.index)
        # Set initial capital
        portfolio['capital'] = capital

        # Iterate through each time period starting from second entry
        for i in range(1, len(portfolio)):
            date = portfolio.index[i]
            prev_date = portfolio.index[i - 1]

            # Generate trading signals using data window up to current date
            self.generate_signals(self.__data__.loc[prev_date:date])

            # Start with previous day's capital
            portfolio.loc[date, 'capital'] = portfolio.loc[prev_date, 'capital']

            # Process each ticker in our trading universe
            for ticker in self.tickers:
                price_col = f"{ticker}_price"

                # Get current price and portfolio value
                current_price: float = self.__data__.at[date, price_col]
                current_portfolio_value = portfolio.loc[date, 'capital']

                # Calculate position size based on signals and current state
                position_size = self.calculate_position_size(
                    ticker,
                    current_price,
                    current_portfolio_value
                )

                # Calculate trade cost and update portfolio value
                cost = position_size * current_price
                portfolio.loc[date, 'capital'] -= cost

                # Track number of trades executed
                if cost != 0:
                    self.__trade_count__ += 1

        return portfolio

    @DeprecationWarning
    def calculate_metrics(self, portfolio: DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics based on Hidden Markov Model.

        This function is now deprecated.
        """

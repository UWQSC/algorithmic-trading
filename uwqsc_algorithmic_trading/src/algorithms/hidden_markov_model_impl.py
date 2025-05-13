"""
Implementation of the Hidden Markov Model (HMM) algorithm.
"""

import random
from typing import Dict, List, Any

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
        name = "Hidden Markov Model"
        data_processor = HMMPreProcessorImpl()

        self.__current_short__: Dict[str, int] = {}
        self.__current_long__: Dict[str, int] = {}
        self.__previous_short__: Dict[str, int] = {}
        self.__previous_long__: Dict[str, int] = {}

        super().__init__(name, tickers, data_processor, parameters)

    def generate_signals(self, current_data: DataFrame):
        # If this is the first time executing, initialize current and previous signals
        #   by looping through each stock, storing its long and short-term values
        if not self.executing:
            for ticker in self.tickers:
                short_col = f"{ticker}_short"
                long_col = f"{ticker}_long"

                self.__current_short__[ticker] = current_data[short_col].iloc[-1]
                self.__current_long__[ticker] = current_data[long_col].iloc[-1]

                # Randomly assign a position to each stock
                self.__positions__[ticker] = random.choice(list(StockPosition))
            self.executing = True
        # If it is not the first run
        else:
            for ticker in self.tickers:
                short_col = f"{ticker}_short"
                long_col = f"{ticker}_long"

                # Save the current short/long values as "previous" for comparison
                # Update the current short/long values
                self.__previous_short__[ticker] = self.__current_short__[ticker]
                self.__previous_long__[ticker] = self.__current_long__[ticker]
                self.__current_short__[ticker] = current_data[short_col].iloc[-1]
                self.__current_long__[ticker] = current_data[long_col].iloc[-1]

                # If the short-term crosses above the long-term, set position to LONG (Buy)
                # If the short-term crosses below the long-term, set position to SHORT (Sell)
                # Else, NEUTRAL (Hold)
                # This is a simple crossover strategy
                if (self.__previous_short__[ticker] <= self.__previous_long__[ticker] and
                        self.__current_short__[ticker] > self.__current_long__[ticker]):
                    self.__positions__[ticker] = StockPosition.LONG
                elif (self.__previous_short__[ticker] > self.__previous_long__[ticker] and
                        self.__current_short__[ticker] <= self.__current_long__[ticker]):
                    self.__positions__[ticker] = StockPosition.SHORT
                else:
                    self.__positions__[ticker] = StockPosition.NEUTRAL

    # Decides how many shares to buy or sell
    # Arguments:
    #  - ticker: The stock ticker symbol
    #  - price: The current price of the stock
    #  - portfolio_value: The current value of the portfolio
    #  - base_position_size: float = self.parameters['position_size'] * portfolio_value
    #       start with the % of the portfolio value you want to use (like 10% of $10,000 = $1,000)
    # - position_size: float = 0.0
    #       default is to not buy anything
    def calculate_position_size(self, ticker: str, price: float, portfolio_value: float) -> float:
        base_position_size: float = self.parameters['position_size'] * portfolio_value
        position_size: float = 0.0

        # If LONG, calculate how many shares to buy with that size and company
        if self.__positions__[ticker] == StockPosition.LONG:
            position_size = base_position_size / price
        # If SHORT, buy a negative number of shares (sell) with that size and company
        #   making it a negative number is a simple way to track sells and buys
        elif self.__positions__[ticker] == StockPosition.SHORT:
            position_size = -1 * (base_position_size / price)

        return position_size

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

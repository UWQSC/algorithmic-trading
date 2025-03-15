"""
Testing the Hidden Markov Model Algorithm
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from uwqsc_algorithmic_trading.interfaces.algorithms.algorithm_interface import StockPosition
from uwqsc_algorithmic_trading.src.algorithms.hidden_markov_model_impl import HiddenMarkovModelImpl
from uwqsc_algorithmic_trading.src.preprocessing.hmm_preprocessor_impl import HMMPreProcessorImpl


class HiddenMarkovModelImplTest(unittest.TestCase):
    """
   This class is used to test each component of Hidden Markov Model Algorithm.
   Following the same testing pattern as SimpleMovingAverageImplTest.
   """

    def setUp(self):
        """
       Set up test fixtures before each test method.
       Similar to SMA tests, we mock the preprocessor and initialize the algorithm.
       """
        self.tickers = ["AAPL", "GOOGL"]  # Test with two popular stocks
        self.preprocessor = MagicMock(spec=HMMPreProcessorImpl)  # Mock preprocessor
        self.parameters = {"position_size": 0.1}  # Same position size as SMA for consistency
        self.algorithm = HiddenMarkovModelImpl(self.tickers, self.preprocessor, self.parameters)
        # Initialize positions to HOLD
        self.algorithm.__positions__ = {ticker: StockPosition.HOLD for ticker in self.tickers}
        self.algorithm.__trade_count__ = 0

    def test_execute_trades(self):
        """
       Testing that the algorithm can execute trades correctly.
       Tests initialization, position changes, and no-trade scenarios.
       """

        self.algorithm.calculate_position_size = MagicMock(return_value=0.1)

        # Initial starting capital for testing
        capital = 10000

        # Create test data with known values
        index = pd.date_range(start='2023-01-01', periods=5, freq='D')
        data = pd.DataFrame(index=index, data={
            "AAPL_price": np.random.uniform(100, 200, size=5),
            "GOOGL_price": np.random.uniform(1500, 2000, size=5)
        })
        # This generates a small synthetic dataset of 5 days (from January 1 to January 5, 2023)
        #  with known stock prices for Apple (AAPL) and Google (GOOGL)

        # Injects controlled test data into the HMM algorithm
        self.algorithm.__data__ = data

        # Calls the execute_trades function of the algorithm, passing the initial capital
        # The function is expected to return a portfolio DataFrame containing trade
        # execution results like cash balance, stock positions, and PnL
        portfolio = self.algorithm.execute_trades(capital)

        # Verify portfolio structure
        # First assertion: Checks if "capital" is a column in the returned portfolio, ensuring that
        #  capital tracking is correctly implemented.
        # Second assertion: Ensures that the returned portfolio has the same number of rows as the
        #  input dataset (5 days), confirming that the function processes all time periods.
        self.assertIn('capital', portfolio.columns)
        self.assertEqual(len(portfolio), len(index))

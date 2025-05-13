"""
Testing the Hidden Markov Model Algorithm
"""

import datetime as dt
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
        self.algorithm = HiddenMarkovModelImpl(self.tickers, self.parameters)
        # Initialize positions to HOLD
        self.algorithm.__positions__ = {ticker: StockPosition.HOLD for ticker in self.tickers}
        self.algorithm.__trade_count__ = 0

    def test_generate_signals_with_not_executing(self):
        """
        Testing that the signals are generated correctly when the algorithm was not executing
        """

        data = pd.DataFrame({
            "AAPL_short": [100, 105],
            "AAPL_long": [102, 103],
            "GOOGL_short": [1500, 1495],
            "GOOGL_long": [1498, 1497]
        })

        self.algorithm.executing = False
        self.algorithm.generate_signals(data)
        self.assertTrue(isinstance(self.algorithm.__positions__["AAPL"], StockPosition))
        self.assertTrue(isinstance(self.algorithm.__positions__["GOOGL"], StockPosition))

    def test_generate_signals_with_already_executing(self):
        """
        Testing that the signals are generated correctly when the algorithm was already executing
        """

        previous_data = pd.DataFrame({
            "AAPL_short": [100],
            "AAPL_long": [102],
            "GOOGL_short": [1500],
            "GOOGL_long": [1498]
        })
        self.algorithm.generate_signals(previous_data)

        current_data = pd.DataFrame({
            "AAPL_short": [105],
            "AAPL_long": [103],
            "GOOGL_short": [1495],
            "GOOGL_long": [1497]
        })
        self.algorithm.generate_signals(current_data)

        self.assertEqual(self.algorithm.__positions__["AAPL"], StockPosition.LONG)
        self.assertEqual(self.algorithm.__positions__["GOOGL"], StockPosition.SHORT)

    def test_calculate_position_size_with_hold(self):
        """
        Testing that the position size is calculated correctly.
        It should return 0 on hold
        """

        price = 100
        portfolio_value = 10000
        ticker = "AAPL"

        self.algorithm.__positions__[ticker] = StockPosition.HOLD
        position_size = self.algorithm.calculate_position_size(ticker, price, portfolio_value)
        self.assertEqual(position_size, 0)

    def test_calculate_position_size_with_short(self):
        """
        Testing that the position size is calculated correctly.
        It should return a -ve value on short
        """

        price = 100
        portfolio_value = 10000
        ticker = "AAPL"

        self.algorithm.__positions__[ticker] = StockPosition.SHORT
        position_size = self.algorithm.calculate_position_size(ticker, price, portfolio_value)
        expected_size = -1 * (self.parameters["position_size"] * portfolio_value) / price
        self.assertEqual(position_size, expected_size)

    def test_calculate_position_size_with_long(self):
        """
        Testing that the position size is calculated correctly.
        It should return a +ve value on short
        """

        price = 100
        portfolio_value = 10000
        ticker = "AAPL"

        self.algorithm.__positions__[ticker] = StockPosition.LONG
        position_size = self.algorithm.calculate_position_size(ticker, price, portfolio_value)
        expected_size = (self.parameters["position_size"] * portfolio_value) / price
        self.assertEqual(position_size, expected_size)

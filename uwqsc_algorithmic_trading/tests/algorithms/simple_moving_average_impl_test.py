"""
Testing the Simple Moving Average Algorithm
"""

import unittest
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from uwqsc_algorithmic_trading.interfaces.algorithms.algorithm_interface import StockPosition
from uwqsc_algorithmic_trading.src.algorithms.simple_moving_average_impl \
    import SimpleMovingAverageImpl
from uwqsc_algorithmic_trading.src.preprocessing.sma_preprocessor_impl import SMAPreProcessorImpl


class SimpleMovingAverageImplTest(unittest.TestCase):
    """
    This class is used to test each component of Simple Moving Average Algorithm
    """

    def setUp(self):
        self.tickers = ["AAPL", "GOOGL"]
        self.preprocessor = MagicMock(spec=SMAPreProcessorImpl)
        self.parameters = {"position_size": 0.1}
        self.algorithm = SimpleMovingAverageImpl(self.tickers, self.preprocessor, self.parameters)
        self.algorithm.__positions__ = {ticker: StockPosition.HOLD for ticker in self.tickers}
        self.algorithm.__trade_count__ = 0

    def test_generate_signals_with_not_executing(self):
        """
        Testing that the signals are generated correctly when the algorithm was not executing
        """

        data = pd.DataFrame({
            "AAPL_sma_short_window": [100, 105],
            "AAPL_sma_long_window": [102, 103],
            "GOOGL_sma_short_window": [1500, 1495],
            "GOOGL_sma_long_window": [1498, 1497]
        })

        self.algorithm.executing = False
        self.algorithm.generate_signals(data)
        self.assertTrue(isinstance(self.algorithm.__positions__["AAPL"], StockPosition))
        self.assertTrue(isinstance(self.algorithm.__positions__["GOOGL"], StockPosition))

    def test_generate_signals_with_already_executing(self):
        """
        Testing that the signals are generated correctly when the algorithm was already executing
        """

        data = pd.DataFrame({
            "AAPL_sma_short_window": [100, 105],
            "AAPL_sma_long_window": [102, 103],
            "GOOGL_sma_short_window": [1500, 1495],
            "GOOGL_sma_long_window": [1498, 1497]
        })

        self.algorithm.executing = True
        self.algorithm.generate_signals(data)
        self.assertEqual(self.algorithm.__positions__["AAPL"], StockPosition.LONG)
        self.assertEqual(self.algorithm.__positions__["GOOGL"], StockPosition.SHORT)

    def test_generate_signals_with_not_short_columns_holds(self):
        """
        Testing that the signals are generated correctly when the algorithm was not executing
        """

        data = pd.DataFrame({
            "AAPL_sma_long_window": [102, 103],
            "GOOGL_sma_long_window": [1498, 1497]
        })

        self.algorithm.executing = True
        self.algorithm.generate_signals(data)
        self.assertEqual(self.algorithm.__positions__["AAPL"], StockPosition.HOLD)
        self.assertEqual(self.algorithm.__positions__["GOOGL"], StockPosition.HOLD)

    def test_generate_signals_with_not_long_columns_holds(self):
        """
        Testing that the signals are generated correctly when the algorithm was not executing
        """

        data = pd.DataFrame({
            "AAPL_sma_short_window": [100, 105],
            "GOOGL_sma_short_window": [1500, 1495]
        })

        self.algorithm.executing = True
        self.algorithm.generate_signals(data)
        self.assertEqual(self.algorithm.__positions__["AAPL"], StockPosition.HOLD)
        self.assertEqual(self.algorithm.__positions__["GOOGL"], StockPosition.HOLD)

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

    def test_execute_trades(self):
        """
        Testing that the algorithm can execute trades
        """

        capital = 10000
        index = pd.date_range(start='2023-01-01', periods=5, freq='D')
        data = pd.DataFrame(index=index, data={
            "AAPL_price": np.random.uniform(100, 200, size=5),
            "GOOGL_price": np.random.uniform(1500, 2000, size=5)
        })

        self.algorithm.__data__ = data
        portfolio = self.algorithm.execute_trades(capital)

        self.assertIn('capital', portfolio.columns)
        self.assertEqual(len(portfolio), len(index))

    def test_calculate_metrics_without_win_rate(self):
        """
        Testing that the algorithm can calculate metrics when trade count is 0
        """

        index = pd.date_range(start='2023-01-01', periods=5, freq='D')
        portfolio = pd.DataFrame(index=index, data={'capital': [10000, 10200, 10150, 10300, 10500]})
        metrics = self.algorithm.calculate_metrics(portfolio)

        self.assertIn("Total Return", metrics)
        self.assertIn("Annual Return", metrics)
        self.assertIn("Sharpe Ratio", metrics)
        self.assertIn("Max Drawdown", metrics)
        self.assertIn("Trade Count", metrics)
        self.assertIn("Win Rate", metrics)

    def test_calculate_metrics_with_win_rate(self):
        """
        Testing that the algorithm can calculate metrics when trade count is not 0
        """

        self.algorithm.__trade_count__ = 1
        self.test_calculate_metrics_without_win_rate()

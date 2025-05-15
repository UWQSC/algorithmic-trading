"""
Testing the Hidden Markov Model Algorithm
"""

import unittest
from unittest.mock import MagicMock

import pandas as pd
import numpy as np

from uwqsc_algorithmic_trading.interfaces.algorithms.algorithm_interface import StockPosition
from uwqsc_algorithmic_trading.src.algorithms.hidden_markov_model_impl import HiddenMarkovModelImpl


class HiddenMarkovModelImplTest(unittest.TestCase):
    """
    This class is used to test each component of Hidden Markov Model Algorithm.
    """

    def setUp(self):
        """
        Set up test fixtures before each test method.
        Mock the preprocessor and initialize the algorithm.
        """
        self.tickers = ["AAPL", "GOOG"]
        self.parameters = {"position_size": 0.1}

        # Mock the preprocessor
        self.mock_preprocessor = MagicMock()
        self.algorithm = HiddenMarkovModelImpl(self.tickers, self.parameters)
        self.algorithm.data_processor = self.mock_preprocessor

        # Mock data for testing
        self.data = pd.DataFrame({
            "AAPL_price": [150, 152, 151, 153],
            "GOOG_price": [2800, 2820, 2810, 2830]
        })

    def test_generate_signals_with_not_executing(self):
        """
        Test generate_signals when no valid features are found (e.g., missing columns).
        """
        incomplete_data = pd.DataFrame({
            "MSFT_price": [300, 305, 310, 315]
        })

        with self.assertRaises(ValueError) as context:
            self.algorithm.generate_signals(incomplete_data)

        self.assertIn("No valid features found", str(context.exception))

    def test_generate_signals_with_already_executing(self):
        """
        Test generate_signals when valid data is provided and signals are generated.
        """
        # Mock the HMM model to avoid actual training
        mock_hmm = MagicMock()
        mock_hmm.predict.return_value = [0, 1, 2, 0]
        self.algorithm._HiddenMarkovModelImpl__train_hmm = MagicMock(return_value=mock_hmm)

        self.algorithm.generate_signals(self.data)

        for ticker in self.tickers:
            self.assertIn(ticker, self.algorithm._positions)
            self.assertIn(self.algorithm._positions[ticker],
                          [StockPosition.LONG, StockPosition.SHORT, StockPosition.HOLD])

    def test_generate_signals_sets_hold_for_neutral_state(self):
        """
        Test that market state 2 results in HOLD signal for all tickers.
        """
        mock_hmm = MagicMock()
        mock_hmm.predict.return_value = [2, 2, 2, 2]  # Final state is 2 = HOLD
        self.algorithm._HiddenMarkovModelImpl__train_hmm = MagicMock(return_value=mock_hmm)

        self.algorithm.generate_signals(self.data)

        for ticker in self.tickers:
            self.assertEqual(self.algorithm._positions[ticker], StockPosition.HOLD)

    def test_generate_signals_with_flat_prices(self):
        """
        Test generate_signals with flat price data (no change).
        Ensures it does not crash or return NaN and assigns valid signals.
        """
        flat_data = pd.DataFrame({
            "AAPL_price": [100, 100, 100, 100],
            "GOOG_price": [2000, 2000, 2000, 2000]
        })

        mock_hmm = MagicMock()
        mock_hmm.predict.return_value = [0, 1, 2, 1]  # Some arbitrary output
        self.algorithm._HiddenMarkovModelImpl__train_hmm = MagicMock(return_value=mock_hmm)

        self.algorithm.generate_signals(flat_data)

        for ticker in self.tickers:
            self.assertIn(self.algorithm._positions[ticker],
                        [StockPosition.LONG, StockPosition.SHORT, StockPosition.HOLD])

    def test_calculate_position_size_with_hold(self):
        """
        Test calculate_position_size when the position is HOLD (should return 0).
        """
        self.algorithm._positions = {
            "AAPL": StockPosition.HOLD
        }
        position_size = self.algorithm.calculate_position_size("AAPL", 150, 10000)
        self.assertEqual(position_size, 0)

    def test_calculate_position_size_with_short(self):
        """
        Test calculate_position_size when the position is SHORT.
        """
        self.algorithm._positions = {
            "AAPL": StockPosition.SHORT
        }
        position_size = self.algorithm.calculate_position_size("AAPL", 150, 10000)
        expected_position_size = -1 * (self.parameters["position_size"] * 10000 / 150)
        self.assertAlmostEqual(position_size, expected_position_size)

    def test_calculate_position_size_with_long(self):
        """
        Test calculate_position_size when the position is LONG.
        """
        self.algorithm._positions = {
            "AAPL": StockPosition.LONG
        }
        position_size = self.algorithm.calculate_position_size("AAPL", 150, 10000)
        expected_position_size = self.parameters["position_size"] * 10000 / 150
        self.assertAlmostEqual(position_size, expected_position_size)


if __name__ == "__main__":
    unittest.main()
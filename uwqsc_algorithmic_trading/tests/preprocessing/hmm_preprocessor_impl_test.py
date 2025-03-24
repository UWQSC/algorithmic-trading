"""
Testing the Hidden Markov Model Preprocessor
"""

import unittest
import pandas as pd

from uwqsc_algorithmic_trading.src.preprocessing.hmm_preprocessor_impl import HMMPreProcessorImpl


class HMMPreprocessorImplTest(unittest.TestCase):
    """
    This class is used to test each component of Hidden Markov Model's preprocessor
    """

    def setUp(self):
        self.tickers = ["AAPL", "GOOGL"]
        self.short_window = 50
        self.long_window = 200

        self.preprocessor = HMMPreProcessorImpl()

    def test_remove_duplicate_timestamps_removes_duplicates_with_tickers(self):
        """
        Test that duplicates are removed based on both 'Date' and 'Stock_Ticker' columns
        """
        # Create a DataFrame with duplicate entries for the same date but different tickers
        # and duplicates with both same date and ticker
        data = pd.DataFrame({
            'Date': ['2025-03-13', '2025-03-13', '2025-03-13', '2025-03-14', '2025-03-14'],
            'Stock_Ticker': ['AAPL', 'GOOGL', 'AAPL', 'AAPL', 'AAPL'],
            'Price': [10, 12, 11, 20, 21]
        })

        # Inject the data into the preprocessor
        self.preprocessor.__processed_data__ = data

        # Call the function under test
        self.preprocessor.remove_duplicate_timestamps()

        # The resulting DataFrame should have 3 rows:
        # (2025-03-13, AAPL), (2025-03-13, GOOGL), (2025-03-14, AAPL)
        processed_data = self.preprocessor.__processed_data__
            
        self.assertEqual(len(processed_data), 3)
        self.assertEqual(len(processed_data[processed_data['Date'] == '2025-03-13']), 2)
        self.assertEqual(len(processed_data[processed_data['Date'] == '2025-03-14']), 1)
"""
Testing the Hidden Markov Model Preprocessor
"""

from unittest.mock import MagicMock
import unittest
import datetime as dt
import pandas as pd

from uwqsc_algorithmic_trading.src.preprocessing.hmm_preprocessor_impl import HMMPreProcessorImpl

class HMMPreprocessorImplTest(unittest.TestCase):
    """
    This class is used to test each component of Hidden Markov Model's preprocessor
    """

    def test_missing_values_extrapolates_na_values(self):
        """
        Test that missing values are correctly handled by missing_values().
        """

        self.tickers = ["AAPL"]
        self.end_date = dt.datetime.now().strftime("%Y-%m-%d")
        self.start_date = (dt.datetime.now() - dt.timedelta(days=365 * 2)).strftime("%Y-%m-%d")

        self.preprocessor = MagicMock(spec=HMMPreProcessorImpl)

        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        price_data = [100 + i for i in range(len(dates))]

        self.preprocessor.__processed_data__ = pd.DataFrame(
            {self.tickers[0]: price_data}, index=dates
        )

        self.preprocessor.__processed_data__.iloc[5] = None
        self.preprocessor.__processed_data__.iloc[15] = None

        self.preprocessor.missing_values()

        self.assertFalse(self.preprocessor.__processed_data__.isna().any().any())


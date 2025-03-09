"""
Testing the Hidden Markov Model Preprocessor
"""

import unittest
import datetime as dt
import pandas as pd

class HMMPreprocessorImplTest(unittest.TestCase):
    """
    This class is used to test each component of Hidden Markov Model's preprocessor
    """

    def test_missing_values_extrapolates_na_values(self):
        """
        Testing that missing values are correctly handled
        """
        
    dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
    price_data = [100 + i for i in range(len(dates))] 

    self.preprocessor.__processed_data__ = pd.DataFrame(
        {self.tickers[0]: price_data}, index=dates
    )

    self.preprocessor.__processed_data__.iloc[5] = None
    self.preprocessor.__processed_data__.iloc[15] = None

     self.assertTrue(self.preprocessor.__processed_data__.isna().any().any())

    self.preprocessor.missing_values()

    self.assertFalse(self.preprocessor.__processed_data__.isna().any().any())


"""
Testing the Hidden Markov Model Preprocessor
"""

import unittest


class HMMPreprocessorImplTest(unittest.TestCase):
    """
    This class is used to test each component of Hidden Markov Model's preprocessor
    """

    def test_missing_values_extrapolates_na_values(self):
        """
        Testing that missing values are correctly handled
        """
        
        self.preprocessor.load_data()
        self.preprocessor.__processed_data__.loc[self.start_date] = [None for _ in self.tickers]
        self.preprocessor.missing_values()
        is_na_in_data = self.preprocessor.__processed_data__.isna().any()
        is_na_in_data = is_na_in_data.to_list()
        self.assertFalse(is_na_in_data[0])

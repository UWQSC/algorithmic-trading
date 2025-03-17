import pandas as pd
import unittest
import datetime as dt
import pandas as pd

from uwqsc_algorithmic_trading.src.preprocessing.hmm_preprocessor_impl import HMMPreProcessorImpl

class HMMPreprocessorImplTest(unittest.TestCase):
    """
    This class is used to test each component of Hidden Markov Model's preprocessor
    """
    def setUp(self):
        self.tickers = ["AAPL"]
        self.end_date = dt.datetime.now().strftime("%Y-%m-%d")
        self.start_date = (dt.datetime.now() - dt.timedelta(days=365 * 2)).strftime("%Y-%m-%d")
        self.short_window = 2
        self.long_window = 3

        self.preprocessor = HMMPreProcessorImpl(self.tickers,
                                                self.start_date,
                                                self.end_date,
                                                self.short_window,
                                                self.long_window)

    def test_remove_duplicate_timestamps_removes_duplicates(self):
        # Create a DataFrame with duplicate 'Date' entries.
        data = pd.DataFrame({
            'Date': ['2025-03-13', '2025-03-13', '2025-03-14', '2025-03-15', '2025-03-15'],
            'Price': [10, 10, 20, 30, 30]
        })

        # Inject the dummy data into the preprocessor.
        # Note: __raw_data__ and __processed_data__ are defined in the base class and we access them via name mangling.
        self.preprocessor._IPreProcessData__raw_data__ = data
        self.preprocessor._IPreProcessData__processed_data__ = data.copy()

        raw_size = self.preprocessor._IPreProcessData__raw_data__.size

        # Call the function under test.
        self.preprocessor.remove_duplicate_timestamps()

        new_size = self.preprocessor._IPreProcessData__processed_data__.size

        # Assert that the size is reduced after duplicates are removed.
        self.assertTrue(new_size < raw_size)    

    def test_missing_values_extrapolates_na_values(self):
        """
        Test that missing values are correctly handled by missing_values().
        """
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        price_data = [100 + i for i in range(len(dates))]

        self.preprocessor.__processed_data__ = pd.DataFrame(
            {self.tickers[0]: price_data}, index=dates
        )

        self.preprocessor.__processed_data__.iloc[5] = None
        self.preprocessor.__processed_data__.iloc[15] = None

        # Ensure missing values exist before processing
        self.assertTrue(self.preprocessor.__processed_data__.isna().any().any())

        # Call the missing values function
        self.preprocessor.missing_values()

        # Assert that missing values have been handled (extrapolated)
        self.assertFalse(self.preprocessor.__processed_data__.isna().any().any())

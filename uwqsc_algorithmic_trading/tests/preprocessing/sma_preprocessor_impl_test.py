"""
Testing the Simple Moving Average Preprocessor
"""

import unittest
import datetime as dt
import pandas as pd

from uwqsc_algorithmic_trading.src.preprocessing.sma_preprocessor_impl import SMAPreProcessorImpl


class SMAPreprocessorImplTest(unittest.TestCase):
    """
    This class is used to test each component of Simple Moving Average's preprocessor
    """

    def setUp(self):
        self.tickers = ["AAPL"]
        self.end_date = dt.datetime.now().strftime("%Y-%m-%d")
        self.start_date = (dt.datetime.now() - dt.timedelta(days=365 * 2)).strftime("%Y-%m-%d")
        self.short_window = 10
        self.long_window = 20

        self.preprocessor = SMAPreProcessorImpl(self.tickers,
                                                self.start_date,
                                                self.end_date,
                                                self.short_window,
                                                self.long_window)

    def test_load_data_adds_correct_columns(self):
        """
        Testing that the preprocessor loads data correctly.
        It should save a dataframe that has a price column for the particular ticker.
        """

        self.tickers = ["AAPL", "GOOGL", "MSFT"]

        self.preprocessor = SMAPreProcessorImpl(self.tickers,
                                                self.start_date,
                                                self.end_date,
                                                self.short_window,
                                                self.long_window)
        self.preprocessor.load_data()

        self.assertEqual(len(self.preprocessor.__processed_data__.columns), len(self.tickers))
        for ticker in self.tickers:
            self.assertTrue(f"{ticker}_price" in self.preprocessor.__processed_data__.columns)

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

    def test_remove_duplicate_timestamps_removes_duplicates(self):
        """
        Testing that duplicate timestamps are removed
        """

        self.preprocessor.load_data()
        previous_dataframe_size = self.preprocessor.__processed_data__.size

        self.preprocessor.__processed_data__ = pd.concat(
            [
                self.preprocessor.__processed_data__,
                self.preprocessor.__processed_data__
            ],
            axis=0
        )
        updated_dataframe_size = self.preprocessor.__processed_data__.size
        self.assertNotEqual(previous_dataframe_size, updated_dataframe_size)

        self.preprocessor.remove_duplicate_timestamps()
        duplicate_removed_dataframe_size = self.preprocessor.__processed_data__.size
        self.assertEqual(previous_dataframe_size, duplicate_removed_dataframe_size)
        self.assertFalse(self.preprocessor.__processed_data__.duplicated().any())

    def test_remove_outliers_removes_outliers(self):
        """
        Testing that outliers are removed
        """

        self.preprocessor.load_data()
        changed_dates_values = {
            dt.datetime(1900, 0o1, 0o1): 99999,
            dt.datetime(2100, 0o1, 0o1): -99999
        }

        for ticker in self.tickers:
            for date_value in changed_dates_values.items():
                self.preprocessor.__processed_data__.loc[
                    date_value[0],
                    f"{ticker}_price"
                ] = date_value[1]
        original_size = len(self.preprocessor.__processed_data__)

        self.preprocessor.remove_outliers()

        for ticker in self.tickers:
            for date in changed_dates_values:
                self.assertEqual(
                    self.preprocessor.__processed_data__.get([ticker, date]),
                    None
                )

        self.assertLessEqual(len(self.preprocessor.__processed_data__), original_size)

"""
Before running the Simple Moving Average Algorithm, we must prepare the data for it.
This file contains the logic behind preprocessing data specifically for Simple Moving Average.
"""

from typing import List

import pandas as pd
import yfinance as yf

from interfaces.preprocessing.preprocessor_interface import IPreProcessData


class SMAPreProcessorImpl(IPreProcessData):
    """
    Data preprocessor for the SMA Crossover algorithm.
    """

    def __init__(self,
                 tickers: List[str],
                 start_date: str,
                 end_date: str,
                 short_window: int = 50,
                 long_window: int = 200):
        """
        Initialize the SMA preprocessor.

        :param tickers: List of ticker symbols
        :param start_date: Start date in YYYY-MM-DD format
        :param end_date: End date in YYYY-MM-DD format
        :param short_window: Short-term moving average window in days
        :param long_window: Long-term moving average window in days
        """
        super().__init__()
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.short_window = short_window
        self.long_window = long_window

    def load_data(self):
        self.__raw_data__ = {}

        for ticker in self.tickers:
            data = yf.download(ticker, start=self.start_date, end=self.end_date)
            if not data.empty:
                self.__raw_data__[ticker] = data

        combined_data = pd.DataFrame()
        for ticker, data in self.__raw_data__.items():
            price_data = data.rename(columns={"Open": f"{ticker}_price"})
            price_data = price_data.drop(["Close", "High", "Low", "Volume"], axis=1, level=0)
            combined_data = pd.concat([combined_data, price_data], axis=1)

        self.__processed_data__ = combined_data

    def missing_values(self):
        if self.__processed_data__ is not None:
            self.__processed_data__ = self.__processed_data__.fillna(method='ffill')
            self.__processed_data__ = self.__processed_data__.fillna(method='bfill')

    def remove_duplicate_timestamps(self):
        if self.__processed_data__ is not None:
            self.__processed_data__ = self.__processed_data__.loc[
                ~self.__processed_data__.index.duplicated(keep='first')
            ]

    def remove_outliers(self):
        if self.__processed_data__ is not None:
            for column in self.__processed_data__.columns:
                if '_price' in column:
                    rolling_mean = self.__processed_data__[column].rolling(window=20).mean()
                    rolling_std = self.__processed_data__[column].rolling(window=20).std()

                    z_scores = (self.__processed_data__[column] - rolling_mean) / rolling_std

                    mask = abs(z_scores) > 3
                    self.__processed_data__.loc[mask, column] = rolling_mean[mask]

            for ticker in self.tickers:
                price_col = f"{ticker}_price"
                if price_col in self.__processed_data__.columns:
                    self.__processed_data__[f"{ticker}_sma_short_window"] = (
                        self.__processed_data__[price_col].rolling(self.short_window).mean()
                    )

                    self.__processed_data__[f"{ticker}_sma_long_window"] = (
                        self.__processed_data__[price_col].rolling(self.long_window).mean()
                    )

            self.__processed_data__ = self.__processed_data__.dropna()

"""
Before running the Hidden Markov Model Algorithm, we must prepare the data for it.
This file contains the logic behind preprocessing data specifically for Hidden Markov Model.
"""

import numpy as np

from pandas import DataFrame
from uwqsc_algorithmic_trading.interfaces.preprocessing.preprocessor_interface import (
    IPreProcessData
)


class HMMPreProcessorImpl(IPreProcessData):
    """
    Data preprocessor for the Hidden Markov Model algorithm.
    Focuses on cleaning and preparing financial time series data.
    """

    def __init__(self):
        super().__init__()
        self.outlier_threshold = 3.0

    def missing_values(self):

        if self.__processed_data__ is not None:
            try:
                self.__processed_data__ = self.__processed_data__.fillna(
                    method="ffill")
                self.__processed_data__ = self.__processed_data__.fillna(
                    method="bfill")
            except TypeError:
                self.__processed_data__ = self.__processed_data__.ffill()
                self.__processed_data__ = self.__processed_data__.bfill()

            self.__processed_data__ = self.__processed_data__.interpolate(
                method='linear')

            for col in self.__processed_data__.columns:
                if 'volume' in col.lower():
                    self.__processed_data__[
                        col] = self.__processed_data__[col].fillna(0)
                elif 'price' in col.lower():
                    mean_val = self.__processed_data__[col].mean()
                    self.__processed_data__[col] = self.__processed_data__[
                        col].fillna(mean_val)

    def remove_duplicate_timestamps(self):
        pass

    def remove_outliers(self):
        if self.__processed_data__ is None:
            return

        data = self.__processed_data__.copy()

        price_columns = [col for col in data.columns if 'price' in col.lower()]

        for col in price_columns:
            if data[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                mean_val = data[col].mean()
                std_val = data[col].std()

                if std_val != 0:
                    z_scores = np.abs((data[col] - mean_val) / std_val)

                    outlier_mask = z_scores > self.outlier_threshold

                    if outlier_mask.any():
                        upper_bound = mean_val + \
                            (self.outlier_threshold * std_val)
                        lower_bound = mean_val - \
                            (self.outlier_threshold * std_val)

                        data.loc[data[col] > upper_bound, col] = upper_bound
                        data.loc[data[col] < lower_bound, col] = lower_bound

        self.__processed_data__ = data

    def process_data(self, current_data: DataFrame) -> DataFrame:
        super().process_data(current_data)
        self.__add_technical_indicators__()
        return self.__processed_data__

    def __add_technical_indicators__(self):
        """
        Add basic technical indicators that are useful for HMM.
        This is called automatically during processing.
        """

        if self.__processed_data__ is None:
            return

        data = self.__processed_data__.copy()

        tickers = set()
        for col in data.columns:
            if '_price' in col:
                ticker = col.replace('_price', '')
                tickers.add(ticker)

        for ticker in tickers:
            price_col = f"{ticker}_price"
            volume_col = f"{ticker}_volume"

            if price_col in data.columns:
                if self.__data_history__ is not None:
                    history_prices = self.__data_history__[price_col]

                    if len(history_prices) > 0:
                        current_price = data[price_col].iloc[-1]
                        last_price = history_prices.iloc[-1]

                        if last_price != 0:
                            price_change = (
                                current_price - last_price) / last_price
                            data[f"{ticker}_price_change"] = price_change
                        else:
                            data[f"{ticker}_price_change"] = 0.0
                    else:
                        data[f"{ticker}_price_change"] = 0.0
                else:
                    data[f"{ticker}_price_change"] = 0.0

                if volume_col in data.columns:
                    if self.__data_history__ is not None:
                        history_volume = self.__data_history__[volume_col]

                        if len(history_volume) > 0:
                            current_volume = data[volume_col].iloc[-1]
                            avg_volume = history_volume.mean()

                            if avg_volume != 0:
                                volume_ratio = current_volume / avg_volume
                                data[f"{ticker}_volume_ratio"] = volume_ratio
                            else:
                                data[f"{ticker}_volume_ratio"] = 1.0
                        else:
                            data[f"{ticker}_volume_ratio"] = 1.0
                    else:
                        data[f"{ticker}_volume_ratio"] = 1.0

        self.__processed_data__ = data

"""
Before running the Simple Tensorflow Algorithm, we must prepare the data for it.
This file contains the logic behind preprocessing data specifically for Simple Tensorflow
"""

from uwqsc_algorithmic_trading.interfaces.preprocessing.preprocessor_interface import (
    IPreProcessData
)


class STPreProcessorImpl(IPreProcessData):
    """
    Data preprocessor for the Simple Tensorflow algorithm.
    """

    def remove_duplicate_timestamps(self):
        pass

    def remove_outliers(self):
        pass

    def missing_values(self):
        pass

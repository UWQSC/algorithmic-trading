"""
Before running the Hidden Markov Model Algorithm, we must prepare the data for it.
This file contains the logic behind preprocessing data specifically for Hidden Markov Model.
"""

from uwqsc_algorithmic_trading.interfaces.preprocessing.preprocessor_interface \
    import IPreProcessData


class HMMPreProcessorImpl(IPreProcessData):
    """
    Data preprocessor for the Hidden Markov Model algorithm.
    """

    def load_data(self):
        pass

    def missing_values(self):
        pass

    def remove_duplicate_timestamps(self):
        # Use processed data if available, otherwise fall back to raw data.
        if self.__processed_data__ is not None:
            data = self.__processed_data__
        else:
            data = self.__raw_data__
        
        # Check if data is loaded
        if data is None:
            raise ValueError("Data not loaded. Please ensure load_data() is called before removing duplicate timestamps.")
        
        # Remove duplicates based on the 'timestamp' column
        clean_data = data.drop_duplicates(subset=['Date'])
        self.__processed_data__ = clean_data

    def remove_outliers(self):
        pass

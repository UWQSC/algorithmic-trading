"""
Before running the Hidden Markov Model Algorithm, we must prepare the data for it.
This file contains the logic behind preprocessing data specifically for Hidden Markov Model.
"""

from interfaces.preprocessing.preprocessor_interface import IPreProcessData


class HMMPreProcessorImpl(IPreProcessData):
    """
    Data preprocessor for the Hidden Markov Model algorithm.
    """

    def load_data(self):
        pass

    def missing_values(self):
        pass

    def remove_duplicate_timestamps(self):
        pass

    def remove_outliers(self):
        pass

"""

"""

from uwqsc_algorithmic_trading.interfaces.preprocessing.preprocessor_interface import \
    IPreProcessData


class PairsPreProcessor(IPreProcessData):
    def missing_values(self):
        pass

    def remove_duplicate_timestamps(self):
        pass

    def remove_outliers(self):
        pass

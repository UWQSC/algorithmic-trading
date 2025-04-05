from scipy.stats import zscore
import pandas as pd


class HMMPreProcessorImpl:
    def __init__(self):
        self.__processed_data__ = None

    def load_data(self, df: pd.DataFrame):
        self.__processed_data__ = df.copy()

    def remove_outliers(self, threshold=3.0):
        """
        Removes outliers using Z-score method.
        """
        data = self.__processed_data__
        if data is None:
            raise ValueError("Data not loaded. Please call load_data() first.")

        z_scores = ((data - data.mean()) / data.std()).abs()
        mask = (z_scores < threshold).all(axis=1)
        self.__processed_data__ = data[mask]

    def get_data(self):
        return self.__processed_data__

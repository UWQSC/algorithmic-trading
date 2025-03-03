"""
Implementation of the Hidden Markov Model (HMM) algorithm.
"""

from typing import Dict, List, Any

from pandas import DataFrame

from interfaces.algorithms.algorithm_interface import IAlgorithm
from src.preprocessing.hmm_preprocessor_impl import HMMPreProcessorImpl


class HiddenMarkovModelImpl(IAlgorithm):
    """
    Working logic for Hidden Markov Model (HMM) algorithm.
    """

    def __init__(self,
                 tickers: List[str],
                 data_processor: HMMPreProcessorImpl,
                 parameters: Dict[str, Any] = None):
        name = "Hidden Markov Model"

        super().__init__(name, tickers, data_processor, parameters)

    def generate_signals(self, data: DataFrame):
        pass

    def calculate_position_size(self, ticker: str, price: float, portfolio_value: float) -> float:
        pass

    def execute_trades(self, capital: float) -> DataFrame:
        pass

    def calculate_metrics(self, portfolio: DataFrame) -> Dict[str, float]:
        pass

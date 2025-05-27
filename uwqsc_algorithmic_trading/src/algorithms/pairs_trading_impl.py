"""
File to add logic for the pairs trading system
"""
from typing import List, Dict, Any

import pandas as pd

from uwqsc_algorithmic_trading.interfaces.algorithms.algorithm_interface import IAlgorithm
from uwqsc_algorithmic_trading.src.preprocessing.pairs_preprocessor import PairsPreProcessor


class PairsTradingImpl(IAlgorithm):


    def __init__(self,
                 tickers: List[str],
                 parameters: Dict[str, Any] = None):
        name = "Pairs Trading"
        data_processor = PairsPreProcessor()

        super().__init__(name, tickers, data_processor, parameters)

    def generate_signals(self, current_data: pd.DataFrame):
        pass

    def calculate_position_size(self, ticker: str, price: float, portfolio_value: float) -> float:
        pass

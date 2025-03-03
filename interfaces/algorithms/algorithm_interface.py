"""
This file stores bare-bones information required for any Stock Prediction Algorithm to follow.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from pandas import DataFrame

from interfaces.preprocessing.preprocessor_interface import IPreProcessData
from src.common.config import INTERFACE_NOT_IMPLEMENTED_ERROR


class StockPosition(Enum):
    """
    Enumeration class made to store the information about stock positions.
    Normally, the market can be defined as Bullish, Bearish or Trailing Sideways.
    Similarly, a person's stock position can be defined with hold, short, long, etc.
    """

    SHORT = -1
    HOLD = 0
    LONG = 1


class IAlgorithm(ABC):
    """
    Essential functions shared by all algorithmic trading algorithms.
    """

    def __init__(self,
                 name: str,
                 tickers: List[str],
                 data_processor: IPreProcessData,
                 parameters: Dict[str, Any] = None):
        """
        Initialize a trading algorithm.

        :param name: String. Algorithm name
        :param tickers: List. List of tickers to trade
        :param data_processor: IPreProcessData. Preprocessor instance for algorithm-specific data
        :param parameters: Dictionary. Algorithm-specific parameters
        """

        self.name = name
        self.tickers = tickers
        self.__data_processor__ = data_processor
        self.parameters = parameters or {}
        self.__positions__ = {ticker: StockPosition.HOLD for ticker in tickers}
        self.metrics = {}
        self.__data__: Optional[DataFrame] = None
        self.executing: bool = False
        self.__trade_count__: int = 0

    @abstractmethod
    def generate_signals(self, data: DataFrame):
        """
        Generate trading signals based on processed market data.

        :param data: DataFrame represents the current processed market data

        :side-effect: Changes positions of the algorithm.
        """

        raise INTERFACE_NOT_IMPLEMENTED_ERROR

    @abstractmethod
    def calculate_position_size(self,
                                ticker: str,
                                price: float,
                                portfolio_value: float) -> float:
        """
        Calculate position size for a trade.

        :param ticker: String that represents trading symbol
        :param price: Float that represents current price
        :param portfolio_value: Float that represents current portfolio value

        :returns: Size of the position to be played.
        """

        raise INTERFACE_NOT_IMPLEMENTED_ERROR

    @abstractmethod
    def execute_trades(self, capital: float) -> DataFrame:
        """
        Execute trades based on signals and manage portfolio.

        :param capital: Value of cash allocated to the algorithm

        :returns: DataFrame with portfolio performance
        """

        raise INTERFACE_NOT_IMPLEMENTED_ERROR

    @abstractmethod
    def calculate_metrics(self, portfolio: DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics for the algorithm.

        :param portfolio: Portfolio performance data

        :returns: Dictionary with performance metrics
        """

        raise INTERFACE_NOT_IMPLEMENTED_ERROR

    def prepare_data(self) -> None:
        """
        Prepare data for the algorithm using the linked data processor.
        """

        if self.__data__ is None:
            self.__data__ = self.__data_processor__.process_data()

    def run(self, capital: float) -> Dict[str, Any]:
        """
        Run the algorithm, optionally preparing data first.

        :returns: Dictionary with algorithm results
        """

        self.prepare_data()
        portfolio = self.execute_trades(capital)
        self.metrics = self.calculate_metrics(portfolio)

        results = {
            'signals': self.__positions__,
            'portfolio': portfolio,
            'metrics': self.metrics,
            'data': self.__data__
        }

        print("==============================================")
        print(f"{self.name}:")
        for metric, value in results['metrics'].items():
            print(f"{metric}: {value}")
        print("==============================================")

        return results

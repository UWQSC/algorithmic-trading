"""
TODO
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional

from pandas import DataFrame

from interfaces.preprocessor_interface import IPreProcessData
from src.common.config import INTERFACE_NOT_IMPLEMENTED_ERROR


class StockPosition(Enum):
    """
    TODO
    """

    SHORT = -1
    HOLD = 0
    LONG = 1


class IAlgorithm(ABC):
    """
    TODO
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

    @abstractmethod
    def generate_signals(self) -> List[StockPosition]:
        """
        Generate trading signals based on processed market data.

        :returns: List of trading signals that are associated with each ticker.
        """

        raise INTERFACE_NOT_IMPLEMENTED_ERROR

    @abstractmethod
    def calculate_position_size(self,
                                signal: StockPosition,
                                ticker: str,
                                price: float,
                                portfolio_value: float) -> float:
        """
        Calculate position size for a trade.

        :param signal: StockPosition. (SHORT, HOLD, LONG)
        :param ticker: String. Trading symbol
        :param price: Float. Current price
        :param portfolio_value: Float. Current portfolio value

        :returns: Size of the position to be played.
        """

        raise INTERFACE_NOT_IMPLEMENTED_ERROR

    @abstractmethod
    def execute_trades(self) -> DataFrame:
        """
        Execute trades based on signals and manage portfolio.

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

    def run(self) -> Dict[str, Any]:
        """
        Run the algorithm, optionally preparing data first.

        :returns: Dictionary with algorithm results
        """

        self.prepare_data()
        signals = self.generate_signals()
        portfolio = self.execute_trades()
        self.metrics = self.calculate_metrics(portfolio)

        return {
            'signals': signals,
            'portfolio': portfolio,
            'metrics': self.metrics,
            'data': self.__data__
        }

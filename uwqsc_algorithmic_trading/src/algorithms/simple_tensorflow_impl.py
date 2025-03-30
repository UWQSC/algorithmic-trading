"""
This file sets up logic for a tensorflow based algorithmic trading algorithm.
"""

from typing import List, Dict, Any, Optional

import pandas as pd

from keras import Sequential, Input
from keras.src.layers import Dense, Dropout
from keras.src.optimizers import Adam

from uwqsc_algorithmic_trading.interfaces.algorithms.algorithm_interface import IAlgorithm
from uwqsc_algorithmic_trading.src.preprocessing.st_preprocessor_impl import STPreProcessorImpl


class SimpleTensorFlowImpl(IAlgorithm):
    """
    Sets up a simple tensorflow model.
    This implementation will be trained on pre-existing algorithms.
    No logic is included, as we wish to see comparison with a base level tensorflow model.
    """

    def __init__(self,
                 tickers: List[str],
                 parameters: Dict[str, Any] = None):
        name = "Simple Tensorflow Model"
        processor = STPreProcessorImpl()
        super().__init__(name, tickers, processor, parameters)

        self.__generate_signals_model__: Optional[Sequential] = None
        self.__calculate_position_size_model__: Optional[Sequential] = None
        self.define_model()

    def define_model(self):
        """
        Define and set up the tensorflow models that will predict the outputs of core functions.
        We define two models

        1. __generate_signals_model__:
        Input shape is the size of all tickers.
        Output is processed with tanh function (between -1 to 1) to generate Stock Positions.

        2. __calculate_position_size_model__:
        Input shape is the size of all tickers.
        Output is processed with linear function (-infinity to infinity) to generate Position Size.
        """

        self.__generate_signals_model__ = Sequential()
        self.__calculate_position_size_model__ = Sequential()

        self.__generate_signals_model__.add(Input(shape=(len(self.tickers),)))
        self.__generate_signals_model__.add(Dense(1024, activation='tanh'))
        self.__generate_signals_model__.add(Dropout(0.5))
        self.__generate_signals_model__.add(Dense(units=2048))
        self.__generate_signals_model__.add(Dropout(0.5))
        self.__generate_signals_model__.add(Dense(units=1024))
        self.__generate_signals_model__.add(Dropout(0.5))
        self.__generate_signals_model__.add(Dense(units=512))
        self.__generate_signals_model__.add(Dropout(0.5))
        self.__generate_signals_model__.add(Dense(units=len(self.tickers)))

        self.__calculate_position_size_model__.add(Input(shape=(len(self.tickers),)))
        self.__calculate_position_size_model__.add(Dense(1024, activation='linear'))
        self.__calculate_position_size_model__.add(Dropout(0.5))
        self.__calculate_position_size_model__.add(Dense(units=2048))
        self.__calculate_position_size_model__.add(Dropout(0.5))
        self.__calculate_position_size_model__.add(Dense(units=1024))
        self.__calculate_position_size_model__.add(Dropout(0.5))
        self.__calculate_position_size_model__.add(Dense(units=512))
        self.__calculate_position_size_model__.add(Dropout(0.5))
        self.__calculate_position_size_model__.add(Dense(units=len(self.tickers)))

        self.__generate_signals_model__.compile(
            loss='mean_squared_error',
            optimizer=Adam(),
            metrics=['accuracy']
        )

        self.__calculate_position_size_model__.compile(
            loss='mean_squared_error',
            optimizer=Adam(),
            metrics=['accuracy']
        )

    def generate_signals(self, current_data: pd.DataFrame):
        self.__generate_signals_model__.predict(current_data)

    def calculate_position_size(self, ticker: str, price: float, portfolio_value: float) -> float:
        return self.__calculate_position_size_model__.predict(ticker, price, portfolio_value)

    def train_generate_signals(self):
        """
        Train __generate_signals_model__ model. The updated weights are stored.
        """

    def train_calculate_position_size(self):
        """
        Train __calculate_position_size_model__ model. The updated weights are stored.
        """

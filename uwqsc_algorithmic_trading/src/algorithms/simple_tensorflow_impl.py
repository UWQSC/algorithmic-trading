"""
This file sets up logic for a tensorflow based algorithmic trading algorithm.
"""

from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

from keras import Sequential, Input
from keras.src.layers import Dense, Dropout
from keras.src.optimizers import Adam

from uwqsc_algorithmic_trading.interfaces.algorithms.algorithm_interface import (
    IAlgorithm,
    StockPosition
)
from uwqsc_algorithmic_trading.src.algorithms.simple_moving_average_impl import \
    SimpleMovingAverageImpl
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

        number_of_tickers = len(self.tickers)

        self.__generate_signals_model__ = Sequential()
        self.__calculate_position_size_model__ = Sequential()

        self.__generate_signals_model__.add(Input(shape=(number_of_tickers,)))
        self.__generate_signals_model__.add(Dense(1024, activation='tanh'))
        self.__generate_signals_model__.add(Dropout(0.5))
        self.__generate_signals_model__.add(Dense(units=2048))
        self.__generate_signals_model__.add(Dropout(0.5))
        self.__generate_signals_model__.add(Dense(units=1024))
        self.__generate_signals_model__.add(Dropout(0.5))
        self.__generate_signals_model__.add(Dense(units=512))
        self.__generate_signals_model__.add(Dropout(0.5))
        self.__generate_signals_model__.add(Dense(units=number_of_tickers))

        self.__calculate_position_size_model__.add(Input(shape=(number_of_tickers,)))
        self.__calculate_position_size_model__.add(Dense(1024, activation='linear'))
        self.__calculate_position_size_model__.add(Dropout(0.5))
        self.__calculate_position_size_model__.add(Dense(units=2048))
        self.__calculate_position_size_model__.add(Dropout(0.5))
        self.__calculate_position_size_model__.add(Dense(units=1024))
        self.__calculate_position_size_model__.add(Dropout(0.5))
        self.__calculate_position_size_model__.add(Dense(units=512))
        self.__calculate_position_size_model__.add(Dropout(0.5))
        self.__calculate_position_size_model__.add(Dense(units=number_of_tickers))

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
        observations: List[int] = self.__generate_signals_model__.predict(current_data)
        positions = self.__process_generate_signals_observations__(observations)

        for index, ticker in enumerate(self.tickers):
            self.__positions__[ticker] = positions[index]

    def calculate_position_size(self, ticker: str, price: float, portfolio_value: float) -> float:
        return self.__calculate_position_size_model__.predict(
            price,
            portfolio_value,
            self.__positions__[ticker]
        )

    @staticmethod
    def __process_generate_signals_observations__(obs: List[int]) -> List[StockPosition]:
        """
        Processes the output of __generate_signals_model__ model to get a stock position.

        :param obs: List of integers having an integer for each stock.
        These integers represent the relative stock positions that our model assigns to the stock. 

        :return: A list of Stock Positions for each ticker.
        """

        list_of_positions: List[StockPosition] = []

        for observation in obs:
            position: StockPosition = StockPosition.HOLD

            if -1 <= observation <= -0.34:
                position = StockPosition.SHORT
            elif 0.34 <= observation <= 1:
                position = StockPosition.LONG

            list_of_positions.append(position)

        return list_of_positions

    def __train_generate_signals__(self,
                                   data: pd.DataFrame,
                                   epochs: int,
                                   batch_size: int,
                                   validation_split: float,
                                   verbose: int,
                                   shuffle: bool):
        """
        Train __generate_signals_model__ model. The updated weights are stored.
        """

        mathematical_model: IAlgorithm = SimpleMovingAverageImpl(self.tickers)
        mathematical_model.generate_signals(data)
        position_signals: List[StockPosition] = mathematical_model.__positions__.values()
        actual_signals: List[int] = [int(int_values) for int_values in position_signals]
        data_array = np.array(data)
        actual_signals_array = np.array(actual_signals)

        history = self.__generate_signals_model__.fit(
            data_array,
            actual_signals_array,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            shuffle=shuffle
        )

        print(f"[SIMPLE TENSORFLOW MODEL TRAIN GENERATE SIGNALS] History: ${history}")

    def __train_calculate_position_size__(self,
                                          data: pd.DataFrame,
                                          portfolio_value: int,
                                          epochs: int,
                                          batch_size: int,
                                          validation_split: float,
                                          verbose: int,
                                          shuffle: bool):
        """
        Train __calculate_position_size_model__ model. The updated weights are stored.
        """

        mathematical_model: IAlgorithm = SimpleMovingAverageImpl(self.tickers)
        mathematical_model.generate_signals(data)
        actual_signals: List[float] = []

        for ticker in self.tickers:
            price_col = f"{ticker}_price"
            value: float = data[price_col].iloc[-1]

            actual_signal = mathematical_model.calculate_position_size(
                ticker,
                value,
                portfolio_value
            )
            actual_signals.append(actual_signal)

        data_array = np.array(data)
        actual_signals_array = np.array(actual_signals)

        history = self.__generate_signals_model__.fit(
            data_array,
            actual_signals_array,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            shuffle=shuffle
        )

        print(f"[SIMPLE TENSORFLOW MODEL TRAIN CALCULATE POSITION SIZE] History: ${history}")

    def train(self,
              data: pd.DataFrame,
              capital: int,
              epochs=10,
              batch_size=16,
              validation_split=0.1,
              verbose=1,
              shuffle=False):
        """
        Singular function that trains the entire tensorflow model.
        """

        self.__train_generate_signals__(
            data,
            epochs,
            batch_size,
            validation_split,
            verbose,
            shuffle
        )
        self.__train_calculate_position_size__(
            data,
            capital,
            epochs,
            batch_size,
            validation_split,
            verbose,
            shuffle
        )

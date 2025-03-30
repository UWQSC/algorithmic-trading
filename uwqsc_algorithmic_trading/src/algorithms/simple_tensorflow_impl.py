"""

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
    
    """

    def __init__(self,
                 tickers: List[str],
                 parameters: Dict[str, Any] = None):
        name = "Simple Tensorflow Model"
        processor = STPreProcessorImpl()
        super().__init__(name, tickers, processor, parameters)

        self.__model__: Optional[Sequential] = None
        self.define_model()    

    def define_model(self):
        """
        
        """

        self.__model__ = Sequential()

        self.__model__.add(Input(shape=(len(self.tickers),)))
        self.__model__.add(Dense(128, activation='relu'))
        self.__model__.add(Dropout(0.5))
        self.__model__.add(Dense(units=256))
        self.__model__.add(Dropout(0.5))
        self.__model__.add(Dense(units=128))
        self.__model__.add(Dropout(0.5))
        self.__model__.add(Dense(units=32))
        self.__model__.add(Dropout(0.5))
        self.__model__.add(Dense(units=len(self.tickers)))

        self.__model__.compile(
            loss='mean_squared_error',
            optimizer=Adam(learning_rate=0.001),
            metrics=['accuracy']
        )

    def generate_signals(self, current_data: pd.DataFrame):
        pass

    def calculate_position_size(self, ticker: str, price: float, portfolio_value: float) -> float:
        pass

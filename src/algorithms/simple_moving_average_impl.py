"""
Implementation of the Simple Moving Average (SMA) algorithm.
"""

import random
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from interfaces.algorithms.algorithm_interface import IAlgorithm, StockPosition
from src.preprocessing.sma_preprocessor_impl import SMAPreProcessorImpl


class SimpleMovingAverageImpl(IAlgorithm):
    """
    Working logic for Simple Moving Average (SMA) algorithm.
    """

    def __init__(self,
                 tickers: List[str],
                 data_processor: SMAPreProcessorImpl,
                 parameters: Dict[str, Any] = None):
        name = "Simple Moving Average"

        super().__init__(name, tickers, data_processor, parameters)

    def generate_signals(self, data: pd.DataFrame()):
        if not self.executing:
            for ticker in self.tickers:
                self.__positions__[ticker] = random.choice(list(StockPosition))
            self.executing = True
        else:
            for ticker in self.tickers:
                short_col = f"{ticker}_sma_short_window"
                long_col = f"{ticker}_sma_long_window"

                if short_col not in data.columns or long_col not in data.columns:
                    self.__positions__[ticker] = StockPosition.HOLD
                    continue

                latest_short = data[short_col].iloc[-1]
                latest_long = data[long_col].iloc[-1]
                prev_short = data[short_col].iloc[-2]
                prev_long = data[long_col].iloc[-2]

                if prev_short <= prev_long and latest_short > latest_long:
                    self.__positions__[ticker] = StockPosition.LONG

                elif prev_short >= prev_long and latest_short < latest_long:
                    self.__positions__[ticker] = StockPosition.SHORT

    def calculate_position_size(self, ticker: str, price: float, portfolio_value: float) -> float:
        base_position_size: float = self.parameters['position_size'] * portfolio_value
        position_size: float = 0.0

        if self.__positions__[ticker] == StockPosition.LONG:
            position_size = base_position_size / price
        elif self.__positions__[ticker] == StockPosition.SHORT:
            position_size = -1 * (base_position_size / price)

        return position_size

    def execute_trades(self, capital: float) -> pd.DataFrame:
        portfolio = pd.DataFrame(index=self.__data__.index)
        portfolio['capital'] = capital

        for i in range(1, len(portfolio)):
            date = portfolio.index[i]
            prev_date = portfolio.index[i - 1]
            self.generate_signals(self.__data__.loc[prev_date:date])

            portfolio.loc[date, 'capital'] = portfolio.loc[prev_date, 'capital']

            for ticker in self.tickers:
                price_col = f"{ticker}_price"

                current_price: float = self.__data__.at[date, price_col]
                current_portfolio_value = portfolio.loc[date, 'capital']

                position_size = self.calculate_position_size(
                    ticker,
                    current_price,
                    current_portfolio_value
                )

                cost = position_size * current_price
                portfolio.loc[date, 'capital'] -= cost

                if cost != 0:
                    self.__trade_count__ += 1

        return portfolio

    def calculate_metrics(self, portfolio: pd.DataFrame) -> Dict[str, float]:
        portfolio['daily_return'] = portfolio['capital'].pct_change()

        total_return = (portfolio['capital'].iloc[-1] / portfolio['capital'].iloc[0]) - 1
        annual_return = (1 + total_return) ** (252 / len(portfolio)) - 1

        daily_returns = portfolio['daily_return'].iloc[1:]

        sharpe_ratio = daily_returns.mean() / daily_returns.std() * np.sqrt(252)

        cumulative_returns = (1 + daily_returns).cumprod()
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / running_max) - 1
        max_drawdown = drawdown.min()

        winning_trades = sum(
            1 for i in range(1, len(portfolio)) if portfolio['daily_return'].iloc[i] > 0)
        if self.__trade_count__ > 0:
            win_rate = winning_trades / self.__trade_count__
        else:
            win_rate = 0.0

        return {
            'Total Return': f"{total_return * 100}%",
            'Annual Return': f"{annual_return * 100}%",
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Trade Count': self.__trade_count__,
            'Win Rate': f"{win_rate * 100}%"
        }

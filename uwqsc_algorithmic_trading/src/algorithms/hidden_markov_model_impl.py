"""
Implementation of the Hidden Markov Model (HMM) algorithm.
"""

from typing import Dict, List, Any

from pandas import DataFrame

from uwqsc_algorithmic_trading.interfaces.algorithms.algorithm_interface import IAlgorithm
from uwqsc_algorithmic_trading.src.preprocessing.hmm_preprocessor_impl import HMMPreProcessorImpl

# Import necessary libraries for calculating performance metrics
import numpy as np
from scipy import stats

class HiddenMarkovModelImpl(IAlgorithm):
    """
   Working logic for Hidden Markov Model (HMM) algorithm.
   """

    def __init__(self,
                 tickers: List[str],
                 parameters: Dict[str, Any] = None):
        name = "Hidden Markov Model"
        data_processor = HMMPreProcessorImpl()

        super().__init__(name, tickers, data_processor, parameters)

    def generate_signals(self, current_data: DataFrame):
        pass

    def calculate_position_size(self, ticker: str, price: float, portfolio_value: float) -> float:
        pass

    @DeprecationWarning
    def execute_trades(self, capital: float) -> DataFrame:
        """
       Execute trades based on signals and manage portfolio.
       This implementation follows the same pattern as SimpleMovingAverageImpl.execute_trades
       but uses HMM-specific signals and position calculations.


       :param capital: Value of cash allocated to the algorithm
       :returns: DataFrame with portfolio performance tracking capital changes
       """
        # Initialize portfolio DataFrame with same index as data
        portfolio = DataFrame(index=self.__data__.index)
        # Set initial capital
        portfolio['capital'] = capital

        # Iterate through each time period starting from second entry
        for i in range(1, len(portfolio)):
            date = portfolio.index[i]
            prev_date = portfolio.index[i - 1]

            # Generate trading signals using data window up to current date
            self.generate_signals(self.__data__.loc[prev_date:date])

            # Start with previous day's capital
            portfolio.loc[date, 'capital'] = portfolio.loc[prev_date, 'capital']

            # Process each ticker in our trading universe
            for ticker in self.tickers:
                price_col = f"{ticker}_price"

                # Get current price and portfolio value
                current_price: float = self.__data__.at[date, price_col]
                current_portfolio_value = portfolio.loc[date, 'capital']

                # Calculate position size based on signals and current state
                position_size = self.calculate_position_size(
                    ticker,
                    current_price,
                    current_portfolio_value
                )

                # Calculate trade cost and update portfolio value
                cost = position_size * current_price
                portfolio.loc[date, 'capital'] -= cost

                # Track number of trades executed
                if cost != 0:
                    self.__trade_count__ += 1

        return portfolio

    @DeprecationWarning
    def calculate_metrics(self, portfolio: DataFrame) -> Dict[str, float]:
        """
        Calculate performance metrics based on Hidden Markov Model.
        
        Args:
            portfolio: DataFrame with portfolio performance tracking capital changes
            
        Returns:
            Dict[str, float]: Dictionary of performance metrics
        """
        
        # Ensure we have portfolio data
        if portfolio is None or portfolio.empty:
            return {"error": 1.0, "message": "Empty portfolio data"}
        
        # Calculate daily returns
        portfolio['daily_return'] = portfolio['capital'].pct_change()
        
        # Drop NaN values that occur on the first day
        daily_returns = portfolio['daily_return'].dropna()
        
        # Calculate basic performance metrics
        total_return = (portfolio['capital'].iloc[-1] / portfolio['capital'].iloc[0]) - 1.0
        annualized_return = ((1 + total_return) ** (252 / len(portfolio))) - 1.0
        daily_std = daily_returns.std()
        annualized_volatility = daily_std * np.sqrt(252)
        
        # Calculate Sharpe ratio (assuming risk-free rate of 0 for simplicity)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility != 0 else 0
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + daily_returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()
        
        # Calculate Sortino ratio (downside deviation)
        negative_returns = daily_returns[daily_returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252)
        sortino_ratio = annualized_return / downside_deviation if downside_deviation != 0 else 0
        
        # Calculate HMM-specific metrics
        # 1. State stability: frequency of state changes
        if hasattr(self, '_state_sequence') and len(self._state_sequence) > 1:
            state_changes = sum(1 for i in range(1, len(self._state_sequence)) 
                            if self._state_sequence[i] != self._state_sequence[i-1])
            state_stability = 1.0 - (state_changes / (len(self._state_sequence) - 1))
        else:
            state_stability = 0.0
        
        # 2. Trading efficiency: ratio of profitable trades to total trades
        if self.__trade_count__ > 0:
            trading_efficiency = portfolio['daily_return'][portfolio['daily_return'] > 0].count() / self.__trade_count__
        else:
            trading_efficiency = 0.0
        
        # 3. Calmar ratio: annualized return / maximum drawdown
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # 4. Information ratio (using S&P 500 as benchmark - placeholder)
        # In a real implementation, you would compare to an actual benchmark
        information_ratio = sharpe_ratio  # Simplified for implementation
        
        metrics = {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "annualized_volatility": annualized_volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "information_ratio": information_ratio,
            "state_stability": state_stability,
            "trading_efficiency": trading_efficiency,
            "trade_count": self.__trade_count__,
            "return_skewness": stats.skew(daily_returns),
            "return_kurtosis": stats.kurtosis(daily_returns)
        }
        
        return metrics        

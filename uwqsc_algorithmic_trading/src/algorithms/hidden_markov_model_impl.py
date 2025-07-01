"""
Implementation of the Hidden Markov Model (HMM) algorithm.
"""

from typing import Dict, List, Any, Optional

from pandas import DataFrame
from sklearn.mixture import GaussianMixture

import numpy as np

from uwqsc_algorithmic_trading.interfaces.algorithms.algorithm_interface import (
    IAlgorithm,
    StockPosition
)
from uwqsc_algorithmic_trading.src.preprocessing.hmm_preprocessor_impl import HMMPreProcessorImpl


class HiddenMarkovModelImpl(IAlgorithm):
    """
    Working logic for Hidden Markov Model (HMM) algorithm.
    Uses a 3-state HMM representing Bullish, Bearish, and Sideways market conditions.
    """

    def __init__(self,
                 tickers: List[str],
                 parameters: Optional[Dict[str, Any]] = None):
        name = "Hidden Markov Model"
        data_processor = HMMPreProcessorImpl()
        super().__init__(name, tickers, data_processor, parameters)

        self.__n_states__ = parameters.get('n_states', 3) if parameters else 3
        self.__lookback_window__ = parameters.get(
            'lookback_window', 20) if parameters else 20
        self.__position_threshold__ = parameters.get(
            'position_threshold', 0.6) if parameters else 0.6

        self.__hmm_models__ = {}
        self.__feature_history__ = {}
        self.__state_history__ = {}
        self.__training_features__ = {}

        for ticker in tickers:
            self.__hmm_models__[ticker] = {
                'transition_matrix': self.__initialize_transition_matrix__(),
                'emission_model': GaussianMixture(n_components=self.__n_states__, random_state=42),
                'state_probabilities': np.ones(self.__n_states__) / self.__n_states__,
                'trained': False
            }
            self.__feature_history__[ticker] = []
            self.__state_history__[ticker] = []

    def generate_signals(self, current_data: DataFrame):
        for ticker in self.tickers:
            try:
                features = self.__extract_features__(current_data, ticker)
                self.__update_hmm_model__(ticker, features)
                predicted_state = self.__predict_state__(ticker, features)
                self.__state_history__[ticker].append(predicted_state)

                if len(self.__state_history__[ticker]) > self.__lookback_window__:
                    self.__state_history__[ticker] = self.__state_history__[
                        ticker][-self.__lookback_window__:]

                self.__positions__[ticker] = predicted_state

                if ticker not in self.metrics:
                    self.metrics[ticker] = {}

                model = self.__hmm_models__[ticker]
                if model['trained']:
                    confidence = np.max(model['state_probabilities'])
                    self.metrics[ticker]['last_confidence'] = confidence
                    self.metrics[ticker]['last_state'] = predicted_state

            except Exception as e:
                print(f"Error generating signals for {ticker}: {e}")
                self.__positions__[ticker] = StockPosition.HOLD

    def calculate_position_size(self,
                                ticker: str,
                                price: float,
                                portfolio_value: float) -> float:
        if ticker not in self.__positions__:
            return 0.0

        position = self.__positions__[ticker]
        base_allocation = portfolio_value / len(self.tickers)
        confidence = 1.0
        if ticker in self.metrics and 'last_confidence' in self.metrics[ticker]:
            confidence = self.metrics[ticker]['last_confidence']
        if position == StockPosition.LONG:
            adjusted_allocation = base_allocation * confidence
            position_size = adjusted_allocation / price
        elif position == StockPosition.SHORT:
            adjusted_allocation = base_allocation * confidence
            position_size = -(adjusted_allocation / price)
        else:
            position_size = 0.0

        return position_size

    def get_model_state(self, ticker: str) -> Dict[str, Any]:
        """Get current state information for a ticker."""
        if ticker not in self.__hmm_models__:
            return {}

        model = self.__hmm_models__[ticker]
        state_names = ['Bearish', 'Sideways', 'Bullish']

        return {
            'trained': model['trained'],
            'current_position': self.__positions__.get(ticker, StockPosition.HOLD).name,
            'state_probabilities': dict(zip(state_names, model['state_probabilities'])),
            'last_state': self.__state_history__[ticker][-1] if self.__state_history__[ticker] else None,
            'confidence': self.metrics.get(ticker, {}).get('last_confidence', 0.0)
        }

    def __initialize_transition_matrix__(self) -> np.ndarray:
        """Initialize transition matrix with slight persistence bias."""

        transition_matrix = np.full(
            (self.__n_states__, self.__n_states__), 0.1)
        np.fill_diagonal(transition_matrix, 0.8)

        transition_matrix = transition_matrix / \
            transition_matrix.sum(axis=1, keepdims=True)
        return transition_matrix

    def __extract_features__(self, current_data: DataFrame, ticker: str) -> np.ndarray:
        """Extract features for HMM from current and historical data."""

        price_col = f"{ticker}_price"
        volume_col = f"{ticker}_volume"

        if price_col not in current_data.columns:
            raise ValueError(f"Price column {price_col} not found in data")

        current_price = current_data[price_col].iloc[-1]
        current_volume = current_data[volume_col].iloc[-1] if volume_col in current_data.columns else 0

        history = self.__feature_history__[ticker]

        features = []

        if len(history) > 0:
            prev_price = history[-1][0]
            price_return = (current_price - prev_price) / \
                prev_price if prev_price != 0 else 0
            features.append(price_return)
        else:
            features.append(0.0)

        if len(history) >= 5:
            recent_returns = [h[0]
                              for h in history[-5:]]
            volatility = np.std(recent_returns) if len(
                recent_returns) > 1 else 0
            features.append(volatility)
        else:
            features.append(0.0)

        if len(history) > 0 and len(history[-1]) > 1:
            prev_volume = history[-1][1]
            volume_change = (current_volume - prev_volume) / \
                prev_volume if prev_volume != 0 else 0
            features.append(volume_change)
        else:
            features.append(0.0)

        self.__feature_history__[ticker].append(
            [current_price, current_volume])

        if len(self.__feature_history__[ticker]) > self.__lookback_window__ * 2:
            self.__feature_history__[ticker] = self.__feature_history__[
                ticker][-self.__lookback_window__:]

        return np.array(features).reshape(1, -1)

    def __update_hmm_model__(self, ticker: str, features: np.ndarray):
        """Update HMM model with new observation using online learning approach."""
        model = self.__hmm_models__[ticker]

        if ticker not in self.__training_features__:
            self.__training_features__[ticker] = []

        self.__training_features__[ticker].append(features[0])

        min_samples = max(10, self.__n_states__ * 3)
        if len(self.__training_features__[ticker]) >= min_samples:
            training_data = np.array(self.__training_features__[ticker])

            try:
                model['emission_model'].fit(training_data)
                model['trained'] = True

                if len(self.__training_features__[ticker]) > self.__lookback_window__:
                    self.__training_features__[ticker] = self.__training_features__[
                        ticker][-self.__lookback_window__:]

            except Exception as e:
                print(
                    f"Warning: Failed to train emission model for {ticker}: {e}")

    def __predict_state__(self, ticker: str, features: np.ndarray) -> StockPosition:
        """Predict current market state using HMM."""
        model = self.__hmm_models__[ticker]

        if not model['trained']:
            return StockPosition.HOLD

        try:
            emission_probs = model['emission_model'].predict_proba(features)[0]

            if len(self.__state_history__[ticker]) > 0:
                prev_state = self.__state_history__[ticker][-1]
                transition_probs = model['transition_matrix'][prev_state]

                state_probs = emission_probs * transition_probs
            else:
                state_probs = emission_probs

            if state_probs.sum() > 0:
                state_probs = state_probs / state_probs.sum()
            else:
                state_probs = np.ones(self.__n_states__) / self.__n_states__

            model['state_probabilities'] = state_probs
            predicted_state: StockPosition = StockPosition(
                np.argmax(state_probs))

            max_prob = np.max(state_probs)
            if max_prob < self.__position_threshold__:
                predicted_state = StockPosition.HOLD

            return predicted_state

        except Exception as e:
            print(f"Warning: State prediction failed for {ticker}: {e}")
            return StockPosition.HOLD

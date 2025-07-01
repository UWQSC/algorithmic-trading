"""
Testing the Hidden Markov Model Algorithm
"""

import unittest

import numpy as np
import pandas as pd

from unittest.mock import patch

from uwqsc_algorithmic_trading.src.algorithms.hidden_markov_model_impl import HiddenMarkovModelImpl
from uwqsc_algorithmic_trading.interfaces.algorithms.algorithm_interface import StockPosition

class TestHiddenMarkovModelImpl(unittest.TestCase):
    """Test cases for HiddenMarkovModelImpl class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.tickers = ['AAPL', 'GOOGL']
        self.parameters = {
            'n_states': 3,
            'lookback_window': 20,
            'position_threshold': 0.6
        }
        self.hmm = HiddenMarkovModelImpl(
            tickers=self.tickers, parameters=self.parameters)

    def test_init_basic(self):
        """Test 25: Test basic initialization."""
        hmm = HiddenMarkovModelImpl(tickers=['AAPL'])
        self.assertEqual(hmm.name, "Hidden Markov Model")
        self.assertEqual(hmm.tickers, ['AAPL'])
        self.assertEqual(hmm.__n_states__, 3)
        self.assertEqual(hmm.__lookback_window__, 20)
        self.assertEqual(hmm.__position_threshold__, 0.6)

    def test_init_with_parameters(self):
        """Test 26: Test initialization with custom parameters."""
        params = {'n_states': 4, 'lookback_window': 15,
                  'position_threshold': 0.7}
        hmm = HiddenMarkovModelImpl(tickers=['AAPL'], parameters=params)
        self.assertEqual(hmm.__n_states__, 4)
        self.assertEqual(hmm.__lookback_window__, 15)
        self.assertEqual(hmm.__position_threshold__, 0.7)

    def test_init_hmm_models_created(self):
        """Test 28: Test HMM models are created for each ticker."""
        for ticker in self.tickers:
            self.assertIn(ticker, self.hmm.__hmm_models__)
            self.assertIn('transition_matrix', self.hmm.__hmm_models__[ticker])
            self.assertIn('emission_model', self.hmm.__hmm_models__[ticker])
            self.assertIn('state_probabilities',
                          self.hmm.__hmm_models__[ticker])
            self.assertFalse(self.hmm.__hmm_models__[ticker]['trained'])

    def test_init_feature_history_initialized(self):
        """Test 29: Test feature history is initialized for each ticker."""
        for ticker in self.tickers:
            self.assertIn(ticker, self.hmm.__feature_history__)
            self.assertEqual(self.hmm.__feature_history__[ticker], [])

    def test_init_state_history_initialized(self):
        """Test 30: Test state history is initialized for each ticker."""
        for ticker in self.tickers:
            self.assertIn(ticker, self.hmm.__state_history__)
            self.assertEqual(self.hmm.__state_history__[ticker], [])

    def test_initialize_transition_matrix_shape(self):
        """Test 31: Test transition matrix has correct shape."""
        matrix = self.hmm.__initialize_transition_matrix__()
        self.assertEqual(matrix.shape, (3, 3))

    def test_initialize_transition_matrix_normalized(self):
        """Test 32: Test transition matrix rows sum to 1."""
        matrix = self.hmm.__initialize_transition_matrix__()
        row_sums = matrix.sum(axis=1)
        np.testing.assert_array_almost_equal(row_sums, [1.0, 1.0, 1.0])

    def test_initialize_transition_matrix_diagonal_bias(self):
        """Test 33: Test transition matrix has diagonal bias."""
        matrix = self.hmm.__initialize_transition_matrix__()
        diagonal_values = np.diag(matrix)
        off_diagonal_values = matrix[matrix != np.diag(matrix)]
        self.assertTrue(all(diagonal_values > off_diagonal_values.max()))

    def test_extract_features_missing_price_column(self):
        """Test 34: Test extract_features raises error for missing price column."""
        data = pd.DataFrame({'OTHER_column': [100.0]})

        with self.assertRaises(ValueError) as context:
            self.hmm.__extract_features__(data, 'AAPL')

        self.assertIn("Price column AAPL_price not found",
                      str(context.exception))

    def test_extract_features_no_history(self):
        """Test 35: Test extract_features with no historical data."""
        data = pd.DataFrame({
            'AAPL_price': [100.0],
            'AAPL_volume': [1000]
        })

        features = self.hmm.__extract_features__(data, 'AAPL')

        self.assertEqual(features.shape, (1, 3))
        self.assertEqual(features[0, 0], 0.0)

    def test_extract_features_with_history(self):
        """Test 36: Test extract_features with historical data."""
        self.hmm.__feature_history__['AAPL'] = [[90.0, 900], [95.0, 950]]

        data = pd.DataFrame({
            'AAPL_price': [100.0],
            'AAPL_volume': [1000]
        })

        features = self.hmm.__extract_features__(data, 'AAPL')

        self.assertEqual(features.shape, (1, 3))
        expected_return = (100.0 - 95.0) / 95.0
        self.assertAlmostEqual(features[0, 0], expected_return, places=5)

    def test_extract_features_volatility_calculation(self):
        """Test 37: Test extract_features calculates volatility correctly."""
        price_history = [[90.0, 900], [95.0, 950],
                         [92.0, 920], [98.0, 980], [94.0, 940]]
        self.hmm.__feature_history__['AAPL'] = price_history

        data = pd.DataFrame({
            'AAPL_price': [100.0],
            'AAPL_volume': [1000]
        })

        features = self.hmm.__extract_features__(data, 'AAPL')

        self.assertGreater(features[0, 1], 0)

    def test_extract_features_volume_change(self):
        """Test 38: Test extract_features calculates volume change."""
        self.hmm.__feature_history__['AAPL'] = [[95.0, 950]]

        data = pd.DataFrame({
            'AAPL_price': [100.0],
            'AAPL_volume': [1000]
        })

        features = self.hmm.__extract_features__(data, 'AAPL')

        expected_volume_change = (1000 - 950) / 950
        self.assertAlmostEqual(
            features[0, 2], expected_volume_change, places=5)

    def test_extract_features_memory_management(self):
        """Test 39: Test extract_features manages memory by limiting history."""
        large_history = [[i, i*10]
                         for i in range(50)]
        self.hmm.__feature_history__['AAPL'] = large_history

        data = pd.DataFrame({
            'AAPL_price': [100.0],
            'AAPL_volume': [1000]
        })

        self.hmm.__extract_features__(data, 'AAPL')

        self.assertLessEqual(
            len(self.hmm.__feature_history__['AAPL']), self.hmm.__lookback_window__)

    def test_extract_features_zero_division_protection(self):
        """Test 40: Test extract_features protects against zero division."""
        self.hmm.__feature_history__['AAPL'] = [
            [0.0, 0]]

        data = pd.DataFrame({
            'AAPL_price': [100.0],
            'AAPL_volume': [1000]
        })

        features = self.hmm.__extract_features__(data, 'AAPL')

        self.assertEqual(features[0, 0], 0.0)
        self.assertEqual(features[0, 2], 0.0)

    def test_update_hmm_model_initialization(self):
        """Test 41: Test update_hmm_model initializes training features."""
        features = np.array([[0.1, 0.05, 0.2]])

        self.hmm.__update_hmm_model__('AAPL', features)

        self.assertTrue(hasattr(self.hmm, '__training_features__'))
        self.assertIn('AAPL', self.hmm.__training_features__)

    def test_update_hmm_model_insufficient_data(self):
        """Test 42: Test update_hmm_model doesn't train with insufficient data."""
        features = np.array([[0.1, 0.05, 0.2]])

        self.hmm.__update_hmm_model__('AAPL', features)
        self.assertFalse(self.hmm.__hmm_models__['AAPL']['trained'])

    def test_update_hmm_model_sufficient_data(self):
        """Test 43: Test update_hmm_model trains with sufficient data."""
        self.hmm.__training_features__ = {'AAPL': []}
        for i in range(15):
            features = np.array(
                [[np.random.random(), np.random.random(), np.random.random()]])
            self.hmm.__update_hmm_model__('AAPL', features)

        self.assertTrue(self.hmm.__hmm_models__['AAPL']['trained'])

    def test_update_hmm_model_memory_management(self):
        """Test 44: Test update_hmm_model manages memory by limiting training data."""
        self.hmm.__training_features__ = {
            'AAPL': [np.random.random(3) for _ in range(50)]}

        features = np.array([[0.1, 0.05, 0.2]])
        self.hmm.__update_hmm_model__('AAPL', features)

        self.assertLessEqual(
            len(self.hmm.__training_features__['AAPL']), self.hmm.__lookback_window__)

    @patch('builtins.print')
    def test_update_hmm_model_training_failure(self, mock_print):
        """Test 45: Test update_hmm_model handles training failures gracefully."""
        with patch.object(self.hmm.__hmm_models__['AAPL']['emission_model'], 'fit',
                          side_effect=Exception("Training failed")):

            self.hmm.__training_features__ = {
                'AAPL': [np.random.random(3) for _ in range(15)]}
            features = np.array([[0.1, 0.05, 0.2]])

            self.hmm.__update_hmm_model__('AAPL', features)

            mock_print.assert_called()
            self.assertFalse(self.hmm.__hmm_models__['AAPL']['trained'])

    def test_predict_state_untrained_model(self):
        """Test 46: Test predict_state returns neutral state for untrained model."""
        features = np.array([[0.1, 0.05, 0.2]])

        state = self.hmm.__predict_state__('AAPL', features)

        self.assertEqual(state, StockPosition.HOLD)

    def test_predict_state_trained_model_no_history(self):
        """Test 47: Test predict_state with trained model but no state history."""
        self.hmm.__hmm_models__['AAPL']['trained'] = True
        mock_proba = np.array([[0.1, 0.2, 0.7]])

        with patch.object(self.hmm.__hmm_models__['AAPL']['emission_model'], 'predict_proba',
                          return_value=mock_proba):
            features = np.array([[0.1, 0.05, 0.2]])
            state = self.hmm.__predict_state__('AAPL', features)

            self.assertEqual(state, StockPosition.LONG)

"""
Testing the Hidden Markov Model Preprocessor
"""
import pandas as pd
import unittest


class HMMPreprocessorImplTest(unittest.TestCase):
    """
    This class is used to test each component of Hidden Markov Model's preprocessor
    """
def test_remove_duplicate_timestamps_removes_duplicates(self):

    # Create a DataFrame with duplicate 'Date' entries.
    data = pd.DataFrame({
        'Date': ['2025-03-13', '2025-03-13', '2025-03-14', '2025-03-15', '2025-03-15'],
        'Price': [10, 10, 20, 30, 30]
    })

    # Inject the dummy data into the preprocessor.
    # Note: __raw_data__ and __processed_data__ are defined in the base class and we access them via name mangling.
    self.preprocessor._IPreProcessData__raw_data__ = data
    self.preprocessor._IPreProcessData__processed_data__ = data.copy()

    raw_size = self.preprocessor._IPreProcessData__raw_data__.size

    # Call the function under test.
    self.preprocessor.remove_duplicate_timestamps()

    new_size = self.preprocessor._IPreProcessData__processed_data__.size

    # Assert that the size is reduced after duplicates are removed.
    self.assertTrue(new_size < raw_size)    

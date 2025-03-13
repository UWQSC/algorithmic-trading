"""
Testing the Hidden Markov Model Preprocessor
"""

import unittest


class HMMPreprocessorImplTest(unittest.TestCase):
    """
    This class is used to test each component of Hidden Markov Model's preprocessor
    """
    def test_remove_duplicate_timestamps_removes_duplicates(self):
        self.preprocessor.load_data()
        dataframe_size = self.preprocessor.__raw_data__.size
        self.preprocessor.remove_duplicate_timestamps()
        new_dataframe_size = self.preprocessor.__processed_data__.size
        self.assertTrue(new_dataframe_size < dataframe_size)
        
        
    
    
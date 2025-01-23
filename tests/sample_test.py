"""
Sample test file
"""
import unittest


class SampleTest(unittest.TestCase):
    """
    This class tests sample cases
    """

    def setUp(self):
        """
        This function runs before every unit test.
        """

    def tearDown(self):
        """
        This function runs after every unit test.
        """

    def test_sample(self):
        """
        This function tests sample cases
        """

        self.assertEqual(2 + 2, 4)

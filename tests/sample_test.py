"""
Sample test file
"""

import unittest

from interfaces.sample_interface import ISampleClass
from src.sample_impl import SampleClassImpl


class SampleTest(unittest.TestCase):
    """
    This class tests sample cases
    """

    def setUp(self):
        """
        This function runs before every unit test.
        """
        pass

    def tearDown(self):
        """
        This function runs after every unit test.
        """
        pass

    def test_sample(self):
        """
        This function tests sample cases
        """

        self.assertEqual(2 + 2, 4)

    def test_sample_interface_raises_exception(self):
        """
        Testing our sample implementation
        """
        
        def invalid_function():
            interface_class: ISampleClass = ISampleClass()
        
        self.assertRaises(TypeError, invalid_function)

    def test_sample_implementation_does_not_raise_exception(self):
        """
        Testing our sample implementation
        """
        
        interface_class: ISampleClass = SampleClassImpl()
        self.assertEqual(interface_class.say_hi_to_mom(), "Hi Mom")
        
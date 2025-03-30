"""
This file stores the global constants used in the entire project.
"""

import os.path

# PATH NAMES
class PATHS():
    """
    This class contains all the Paths that are of importance with respect to the AT project.
    """

    COMMON_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_DIR = os.path.dirname(COMMON_DIR)
    PROJECT_DIR = os.path.dirname(SRC_DIR)
    TEST_DIR = os.path.join(PROJECT_DIR, "tests")
    INTERFACE_DIR = os.path.join(PROJECT_DIR, "interface")
    PARENT_DIR = os.path.dirname(PROJECT_DIR)
    DATA_DIR = os.path.join(PARENT_DIR, "data")

# ERRORS
INTERFACE_NOT_IMPLEMENTED_ERROR = RuntimeError("Method Not Implemented")

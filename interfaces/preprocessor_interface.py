"""
This file serves as a standardized interface for data preprocessing and ensures that any subclass
implements this interface which includes preprocessing steps.
"""

from abc import ABC, abstractmethod

class IPreProcessData(ABC):
    """
    This class is an abstract base class meaning it serves as a blueprint for other classes. It is a general class.
    This approach enforces a structured and consistent way to handle preprocessing across different datasets and use cases.

    """

    @abstractmethod
    def load_data(): 
        """ 
	This function saves the DataFrame to Parquet for efficient storage.

	:side-effect: creates a parquet file.
        """

        raise RuntimeError("Method Not Implemented") 


    @abstractmethod
    def missing_values():
        """
        Dataset can contain missing values. These values may be indicated by a null value, or a impossible integer
	(say -99 in a dataset of heights of people in Canada in cm). It is important to remove these missing values
	in cases where having them is harmful for training the model.

:	:side-effect: modifies a parquet file.
	"""

    raise RuntimeError("Method Not Implemented") 

    @abstractmethod
    def remove_duplicate_timestamps():
        """
        Dataset may contain duplicate timestamp values.  It is important to remove the duplicated values
	in cases where having them is harmful for training the model. Note that it is not always required,
	as some indicators benefit from duplications.

	:side-effect: modifies a parquet file.
        """

        raise RuntimeError("Method Not Implemented") 

    @abstractmethod
    def remove_outliers():
        """
	Dataset may contain stocks that have price spikes. These outliers generally skew the training process
	of an algorithm, and should be accounted for when setting up an algorithm. Z-score method will be used
	to remove price spikes.
	
	:side-effect: modifies a parquet file.
        """

        raise RuntimeError("Method Not Implemented") 
    


    



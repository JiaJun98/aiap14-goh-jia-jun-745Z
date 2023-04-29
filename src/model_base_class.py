from abc import ABC, abstractmethod 

"""
This class serves as the abstract base model for all other models to inherit from and
contains common functions to be implemented by the respective classes.
"""
class BaseModel(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train(self):
        """
        Train the model using training data.
        """
        pass

    @abstractmethod
    def predict(self):
        """
        Output prediction results from input data.
        """
        pass
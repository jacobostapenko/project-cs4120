"""
The BaseClassifier class is an abstract class that all classifiers must inherit from. It provides a common interface
By: David Pogrebitskiy and Jacob Ostapenko
Date: 03/31/2023
"""

from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score
import utils


# Abstract class for classifiers
class BaseClassifier(ABC):
    def __init__(self, model):
        self.model = model

    # Abstract methods for training that all classifiers must implement
    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    # Abstract methods for prediction that all classifiers must implement
    @abstractmethod
    def predict(self, X_test):
        pass

    # Method to evaluate the classifier
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return utils.get_prfa(y_test, y_pred)
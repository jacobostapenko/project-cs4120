"""
Logistic Regression model using sklearn inherited from BaseClassifier

By: David Pogrebitskiy and Jacob Ostapenko
Date: 03/31/2023
"""
from sklearn.linear_model import LogisticRegression
from models.BaseClassifier import BaseClassifier


class SklearnClassifier(BaseClassifier):
    def __init__(self, **kwargs):
        super().__init__(LogisticRegression(**kwargs))

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

from pandas.compat import numpy as np


class KNeighborsClassifier:

    def __init__(self, train_X, train_Y) -> None:
        train_X = train_X
        train_Y = train_Y

    def fit(self, X, y):
        pass

    def score(self, X, y):
        pass

    def predict(self, X):
        pass

    def distance(self, x, y):
        return np.linalg.norm(x-y)
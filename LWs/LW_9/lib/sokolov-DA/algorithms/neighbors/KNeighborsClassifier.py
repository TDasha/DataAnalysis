from pandas.compat import numpy as np


class KNeighborsClassifier:

    def __init__(self) -> None:
        super().__init__()

    def fit(self, X, y):
        pass

    def score(self, X, y):
        pass

    def predict(self, X):
        pass

    def distance(self, x, y):
        return np.sqrt(((x - y) ** 2).sum())
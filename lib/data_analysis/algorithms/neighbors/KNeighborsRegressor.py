from .KNeighbors import KNeighbors


class KNeighborsRegressor(KNeighbors):

    def predict(self):
        return self._test_Answers

from .KNeighbors import KNeighbors


class KNeighborsClassifier(KNeighbors):

    def score(self, test_Y):
        counter = [i for i, j in zip(test_Y, self._test_Answers) if i == j]
        return len(counter) / len(test_Y)

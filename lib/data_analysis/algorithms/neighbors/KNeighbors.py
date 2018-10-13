from abc import ABC, abstractmethod


class KNeighbors(ABC):

    def __init__(self, train_X, train_Y, k: int) -> None:
        self.__train_X = train_X
        self.__train_Y = train_Y
        self.__k = k

    def fit(self):
        pass

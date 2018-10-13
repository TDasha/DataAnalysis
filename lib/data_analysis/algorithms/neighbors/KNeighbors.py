from abc import ABC, abstractmethod


class KNeighbors(ABC):

    def __init__(self, train_X, train_Y) -> None:
        train_X = train_X
        train_Y = train_Y

    @abstractmethod
    def fit(self):
        pass

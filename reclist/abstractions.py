from abc import ABC, abstractmethod


class RecDataset(ABC):

    def __init__(self, x_train: list, y_train: list, x_test: list, y_test: list, catalog: dict):
        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test
        self._catalog = catalog

    @property
    def x_train(self):
        return self._x_train

    @property
    def y_train(self):
        return self._y_train

    @property
    def x_test(self):
        return self._x_test

    @property
    def y_test(self):
        return self._y_test

    @property
    def catalog(self):
        return self._catalog


class RecModel(ABC):
    """
    Abstract class for recommendation model
    """

    def __init__(self, model=None):
        self._model = model

    @abstractmethod
    def predict(self, prediction_input: list, *args, **kwargs):
        return NotImplementedError

    @property
    def model(self):
        return self._model


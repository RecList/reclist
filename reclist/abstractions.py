
from datetime import datetime
import os
from abc import ABC, abstractmethod
from functools import wraps
from pathlib import Path
import time
import json
from reclist.utils.train_w2v import train_embeddings
from reclist.current import current
import pandas as pd

class RecDataset(ABC):
    """
    Implements an abstract class for the dataset
    """
    def __init__(self, force_download=False, **kwargs):
        """
        :param force_download: allows to force the download of the dataset in case it is needed.
        :type: force_download: bool, optional
        """
        self._x_train = None
        self._y_train = None
        self._x_test = None
        self._y_test = None
        self._catalog = None
        self.force_download = force_download
        self.load(**kwargs)

    @abstractmethod
    def load(self, **kwargs):
        """
        Abstract method that should implement dataset loading
        :return:
        """
        return

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
        """

        :param model: a model that can be used in the predict function
        """
        self._model = model

    @abstractmethod
    def predict(self, prediction_input: list, *args, **kwargs) -> pd.DataFrame:
        """
        The predict function should implement the behaviour of the model at inference time.

        :param prediction_input: the input that is used to to do the prediction
        :param args:
        :param kwargs:
        :return:
        """
        return NotImplementedError

    @property
    def model(self):
        return self._model


def rec_test(test_type: str):
    """
    Rec test decorator
    """
    def decorator(f):
        @wraps(f)
        def w(*args, **kwargs):
            return f(*args, **kwargs)

        # add attributes to f
        w.is_test = True
        w.test_type = test_type
        try:
            w.test_desc = f.__doc__.lstrip().rstrip()
        except:
            w.test_desc = ""
        try:
            # python 3
            w.name = w.__name__
        except:
            # python 2
            w.name = w.__func__.func_name
        return w

    return decorator


class RecList(ABC):
    META_DATA_FOLDER = '.reclist'

    def __init__(self, model: RecModel, dataset: RecDataset, y_preds: pd.DataFrame = None):
        """
        :param model:
        :param dataset:
        :param y_preds:
        """

        self.name = self.__class__.__name__
        self._rec_tests = self.get_tests()
        self._x_train: pd.DataFrame = dataset.x_train
        self._y_train: pd.DataFrame = dataset.y_train
        self._x_test: pd.DataFrame = dataset.x_test
        self._y_test: pd.DataFrame = dataset.y_test
        self._y_preds =  model.predict(dataset.x_test) if isinstance(y_preds, type(None)) else y_preds
        self.rec_model = model
        self.product_data: pd.DataFrame = dataset.catalog
        self._test_results = []
        self._test_data = {}
        self._dense_repr = {}

        assert len(self._y_test) == len(self._y_preds)

    def train_dense_repr(self, type_name: str, type_fn):
        """
        Train a dense representation over a type of meta-data & store into object
        """
        # type_fn: given a SKU returns some type i.e. brand
        x_train_transformed = [[type_fn(e) for e in session if type_fn(e)] for session in self._x_train]
        wv = train_embeddings(x_train_transformed)
        # store a dict
        self._dense_repr[type_name] = {word: list(wv.get_vector(word)) for word in wv.key_to_index}

    def get_tests(self):
        """
        Helper to extract methods decorated with rec_test
        """

        nodes = {}
        for _ in self.__dir__():
            if not hasattr(self,_):
                continue
            func = getattr(self, _)
            if hasattr(func, 'is_test'):
                nodes[func.name] = func

        return nodes

    def __call__(self, verbose=True, *args, **kwargs):
        run_epoch_time_ms = round(time.time() * 1000)
        # create datastore
        current._report_path = os.path.join(self.META_DATA_FOLDER,
                                        self.name,
                                        self.rec_model.__class__.__name__,
                                        str(run_epoch_time_ms))

        Path(os.path.join(current.report_path, 'artifacts')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(current.report_path, 'results')).mkdir(parents=True, exist_ok=True)
        Path(os.path.join(current.report_path, 'plots')).mkdir(parents=True, exist_ok=True)

        # iterate through tests
        for test_func_name, test in self._rec_tests.items():
            test_result = test(*args, **kwargs)
            # we could store the results in the test function itself
            # test.__func__.test_result = test_result
            self._test_results.append({
                'test_name': test.test_type,
                'description': test.test_desc,
                'test_result': test_result}
            )
            if verbose:
                print("============= TEST RESULTS ===============")
                print("Test Type        : {}".format(test.test_type))
                print("Test Description : {}".format(test.test_desc))
                print("Test Result      : {}\n".format(test_result))
        # at the end, we dump it locally
        if verbose:
            print("Generating reports at {}".format(datetime.utcnow()))
        return self.generate_report(run_epoch_time_ms)

    def generate_report(self, epoch_time_ms: int):
        # create path first: META_DATA_FOLDER / RecList / Model / Run Time
        report_path = os.path.join(
            self.META_DATA_FOLDER,
            self.name,
            self.rec_model.__class__.__name__,
            str(epoch_time_ms)
        )

        # now, dump results
        self.dump_results_to_json(self._test_results, report_path, epoch_time_ms)
        # now, store artifacts
        self.store_artifacts(report_path)
        return report_path

    def store_artifacts(self, report_path: str):
        target_path = os.path.join(current.report_path, 'artifacts')
        # store predictions

        self._x_test.to_parquet(os.path.join(target_path,'x_test.pk'))
        self._y_test.to_parquet(os.path.join(target_path,'y_test.pk'))
        self._y_preds.to_parquet(os.path.join(target_path,'y_preds.pk'))


    def dump_results_to_json(self, test_results: list, report_path: str, epoch_time_ms: int):
        target_path = os.path.join(report_path, 'results')
        # make sure the folder is there, with all intermediate parents
        Path(target_path).mkdir(parents=True, exist_ok=True)
        report = {
            'metadata': {
                'run_time': epoch_time_ms,
                'model_name': self.rec_model.__class__.__name__,
                'reclist': self.name,
                'tests': list(self._rec_tests.keys())
            },
            'data': test_results
        }
        with open(os.path.join(target_path, 'report.json'), 'w') as f:
            json.dump(report, f, indent=2)

    @property
    def test_results(self):
        return self._test_results

    @property
    def test_data(self):
        return self._test_data

    @property
    def rec_tests(self):
        return self._rec_tests

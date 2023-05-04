import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from functools import wraps
from reclogger import logger_factory, LOGGER

class Current:
    def __init__(self):
        self._report_path = None

    @property
    def report_path(self):
        return self._report_path


current = Current()

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
    META_DATA_FOLDER = ".reclist"

    def __init__(self, model, dataset, metadata, logging_parameters=None):
        """
        :param model:
        :param dataset:
        """

        self.name = self.__class__.__name__
        self._rec_tests = self.get_tests()
        self.model = model
        self.dataset = dataset
        self.metadata = metadata
        self._test_results = []

    def get_tests(self):
        """
        Helper to extract methods decorated with rec_test
        """

        nodes = {}
        for _ in self.__dir__():
            if not hasattr(self, _):
                continue
            func = getattr(self, _)
            if hasattr(func, "is_test"):
                nodes[func.name] = func

        return nodes

    def __call__(self, verbose=True, logger=LOGGER.COMET, *args, **kwargs):
        logger = logger_factory(logger)()

        run_epoch_time_ms = round(time.time() * 1000)
        # create datastore
        current._report_path = os.path.join(
            self.META_DATA_FOLDER,
            self.name,
            self.model.__class__.__name__,
            str(run_epoch_time_ms),
        )

        Path(os.path.join(current.report_path, "artifacts")).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(current.report_path, "results")).mkdir(
            parents=True, exist_ok=True
        )
        Path(os.path.join(current.report_path, "plots")).mkdir(
            parents=True, exist_ok=True
        )

        # iterate through tests
        for test_func_name, test in self._rec_tests.items():
            test_result = test(*args, **kwargs)
            # we could store the results in the test function itself
            # test.__func__.test_result = test_result
            self._test_results.append(
                {
                    "test_name": test.test_type,
                    "description": test.test_desc,
                    "test_result": test_result,
                }
            )
            logger.write(test.test_type, test_result)

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
            self.model.__class__.__name__,
            str(epoch_time_ms),
        )

        # now, dump results
        self.dump_results_to_json(self._test_results, report_path, epoch_time_ms)

        # now, store artifacts
        self.store_artifacts(report_path)
        return report_path

    def store_artifacts(self, report_path: str):
        target_path = os.path.join(current.report_path, "artifacts")
        # store predictions

        #self._x_test.to_parquet(os.path.join(target_path, "x_test.pk"))
        #self.get_targets().to_parquet(os.path.join(target_path, "y_test.pk"))
        #self.predict().to_parquet(os.path.join(target_path, "y_preds.pk"))

    def dump_results_to_json(
        self, test_results: list, report_path: str, epoch_time_ms: int
    ):
        target_path = os.path.join(report_path, "results")
        # make sure the folder is there, with all intermediate parents
        Path(target_path).mkdir(parents=True, exist_ok=True)
        report = {
            "metadata": {
                "run_time": epoch_time_ms,
                "model_name": self.model.__class__.__name__,
                "reclist": self.name,
                "tests": list(self._rec_tests.keys()),
            },
            "data": test_results,
        }
        with open(os.path.join(target_path, "report.json"), "w") as f:
            json.dump(report, f, indent=2)

    @property
    def rec_tests(self):
        return self._rec_tests


class SessionRecList(RecList):
    def __init__(self, model, dataset, metadata):
        """
        :param model:
        :param dataset:
        """

        super().__init__(model, dataset, metadata)

    @abstractmethod
    def get_targets(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    @rec_test(test_type="Accuracy")
    def accuracy(self):
        """
        Compute the accuracy
        """
        from sklearn.metrics import accuracy_score

        return accuracy_score(
            self.get_targets(), self.predict()
        )


class CoveoSessionRecList(SessionRecList):

    def __init__(self, model, dataset, metadata, similarity_model):
        """
        :param model:
        :param dataset:
        """

        super().__init__(model, dataset, metadata)
        self.similarity_model = similarity_model

    def predict(self):
        """
        Do something
        """
        return [1, 1, 1, 1]

    def get_targets(self):
        """
        Do something
        """
        return self.dataset

    @rec_test(test_type="SlicedAccuracy")
    def sliced_accuracy(self):
        """
        Compute the accuracy
        """
        from metrics.standard_metrics import accuracy_per_slice
        return accuracy_per_slice(
            self.get_targets(), self.predict(), self.metadata["categories"]
        )

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    print("Dotenv not loaded: if you need ENV variables, make sure you export them manually")


cd = CoveoSessionRecList("", [1, 1, 1, 0],
                         {"categories": ["cat", "cat", "cat", "dog"]}, "")
cd(verbose=True, logger=LOGGER.COMET)

"""

### GLOBAL IMPORT SECTION ###

"""


import json
import os
import time
from enum import Enum
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from functools import wraps


"""

### METADATA STORE SECTION ###

"""


class METADATA_STORE(Enum):
    LOCAL = 1
    S3 = 2


class MetaStore(ABC):

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def write_file(self, path, data, is_json=False):
        pass


def metadata_store_factory(label) -> MetaStore:
    if label == METADATA_STORE.S3:
        raise S3MetaStore
    else:
        return LocalMetaStore


class LocalMetaStore(MetaStore):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def write_file(self, path, data, is_json=False):
        if is_json:
            with open(path, "w") as f:
                json.dump(data, f, indent=2)
        else:
            with open(path, "w") as f:
                f.write(data)

        return


class S3MetaStore(MetaStore):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def write_file(self, path, data, is_json=False):
        """
        We use s3fs to write to s3 - note: credentials are the default one
        locally stored in ~/.aws/credentials

        https://s3fs.readthedocs.io/en/latest/
        """
        import s3fs

        s3 = s3fs.S3FileSystem(anon=False)
        if is_json:
            with s3.open(path, 'wb') as f:
                json.dump(data, f, indent=2)
        else:
            with s3.open(path, 'wb') as f:
                f.write(data)

        return


"""

### LOGGING SECTION ###

"""

class LOGGER(Enum):
    LOCAL = 1
    COMET = 2
    NEPTUNE = 3


class RecLogger(ABC):

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def write(self, label, value):
        pass


def logger_factory(label) -> RecLogger:
    if label == LOGGER.COMET:
        return CometLogger
    elif label == LOGGER.NEPTUNE:
        raise NotImplementedError
    else:
        return LocalLogger


class LocalLogger(RecLogger):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def write(self, label, value):
        from rich import print
        print(f"[italic red]{label}[/italic red]:{value}")

        return


class CometLogger(RecLogger):

    def __init__(self, *args, **kwargs):
        """

        In order of priority, use first the kwargs, then the env variables

        """
        super().__init__(*args, **kwargs)
        from comet_ml import Experiment

        api_key = kwargs["COMET_KEY"] if "COMET_KEY" in kwargs else os.environ["COMET_KEY"]
        project_name = kwargs["COMET_PROJECT_NAME"] if "COMET_PROJECT_NAME" in kwargs else os.environ["COMET_PROJECT_NAME"]
        workspace = kwargs["COMET_WORKSPACE"] if "COMET_WORKSPACE" in kwargs else os.environ["COMET_WORKSPACE"]

        # set up the experiment
        self.experiment = Experiment(
            api_key=api_key,
            project_name=project_name,
            workspace=workspace,
        )

        return

    def write(self, label, value):
        self.experiment.log_metric(label, value)
        return

"""

### RECLIST SECTION ###

"""

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

    # this is the target metadata folder
    # it can be overwritten by the user
    # if an env variable is set
    META_DATA_FOLDER = os.environ.get("RECLIST_META_DATA_FOLDER", ".reclist")

    def __init__(
            self,
            model,
            dataset,
            metadata,
            logger: LOGGER = LOGGER.LOCAL,
            metadata_store: METADATA_STORE = METADATA_STORE.LOCAL,
            **kwargs,
            ):
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
        self.logger = logger
        self.logger_service = logger_factory(logger)(**kwargs)
        # if s3 is used, we need to specify the bucket
        self.metadata_bucket = kwargs["bucket"] if "bucket" in kwargs else None
        assert self.metadata_bucket is not None if metadata_store == METADATA_STORE.S3 else True, "If using S3, you need to specify the bucket"
        self.metadata_store_service = metadata_store_factory(metadata_store)(**kwargs)
        self.metadata_store = metadata_store

        return

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

    def create_data_store(self):
        """
        Each reclist run stores artifacts in

        METADATA_FOLDER/ReclistName/ModelName/RunEpochTimeMs
        """
        run_epoch_time_ms = round(time.time() * 1000)
        # specify a bucket as the root of the datastore if using s3
        bucket = self.metadata_bucket if self.metadata_store == METADATA_STORE.S3 else ''
        # create datastore path
        report_path = os.path.join(
            bucket,
            self.META_DATA_FOLDER,
            self.name,
            self.model.__class__.__name__,
            str(run_epoch_time_ms),
        )
        # create subfolders in the local file system if needed
        folders = ["artifacts", "results", "plots"]
        if self.metadata_store == METADATA_STORE.LOCAL:
            for folder in folders:
                Path(os.path.join(report_path, folder)).mkdir(
                    parents=True, exist_ok=True
                )

        return report_path

    def _display_rich_table(self, table_name: str, results: list):
        from rich.console import Console
        from rich.table import Table
        # build the rich table
        table = Table(title=table_name)
        table.add_column("Type", justify="right", style="cyan", no_wrap=True)
        table.add_column("Description ", style="magenta", no_wrap=False)
        table.add_column("Result", justify="right", style="green")
        for result in results:
            table.add_row(
                result['name'],
                result['description'],
                str(round(result['result'], 4)) # rich needs strings to display
                )
        # print out the table
        console = Console()
        console.print(table)

        return

    def __call__(self, verbose=True, *args, **kwargs):
        from rich.progress import track

        self.report_path = self.create_data_store()
        # iterate through tests
        for test_func_name, test in track(self._rec_tests.items(), description="Running RecTests"):
            test_result = test(*args, **kwargs)
            # we could store the results in the test function itself
            # test.__func__.test_result = test_result
            self._test_results.append(
                {
                    "name": test.test_type,
                    "description": test.test_desc,
                    "result": test_result,
                }
            )
            self.logger_service.write(test.test_type, test_result)
        # finally, display all results in a table
        self._display_rich_table(self.name, self._test_results)
        # at the end, we dump it locally
        return self._generate_report(self._test_results, self.report_path)

    def _generate_report(self, test_results: list, report_path: str):
        """
        Store a copy of the results into a file in the metadata store

        TODO: decide what to do with plots and artifacts
        """
        report_file_name = self._dump_results_to_json(test_results, report_path)
        # TODO: decide how store artifacts / if / where
        # self.store_artifacts(report_path)
        return report_path

    def _dump_results_to_json(self, test_results: list, report_path: str):
        report = {
            "metadata": {
                "model_name": self.model.__class__.__name__,
                "reclist": self.name,
                "tests": list(self._rec_tests.keys()),
            },
            "data": test_results,
        }
        report_file_name = os.path.join(report_path, "results", "report.json")
        self.metadata_store_service.write_file(
            report_file_name,
            report,
            is_json=True
        )

        return report_file_name

    @property
    def rec_tests(self):
        return self._rec_tests

    @abstractmethod
    def get_targets(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class CoveoSessionRecList(RecList):

    def __init__(
        self,
        model,
        dataset,
        metadata,
        logger: LOGGER,
        metadata_store: METADATA_STORE,
        **kwargs
    ):
        super().__init__(
            model,
            dataset,
            metadata,
            logger,
            metadata_store,
            **kwargs
        )
        self.similarity_model = kwargs.get("similarity_model", None)

        return

    def predict(self):
        """
        Do something
        """
        return self.model.predict()

    def get_targets(self):
        """
        Do something
        """
        return self.dataset

    @rec_test(test_type="SlicedAccuracy")
    def sliced_accuracy(self):
        """
        Compute the accuracy by slice
        """
        from metrics.standard_metrics import accuracy_per_slice
        # fake some processing time
        import time
        time.sleep(3)

        return accuracy_per_slice(
            self.get_targets(), self.predict(), self.metadata["categories"]
        )

    @rec_test(test_type="Accuracy")
    def accuracy(self):
        """
        Compute the accuracy
        """
        from sklearn.metrics import accuracy_score
        # fake some processing time
        import time
        time.sleep(3)

        return accuracy_score(
            self.get_targets(), self.predict()
        )

"""

### RUNNING SECTION ###

"""

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    print("Dotenv not loaded: if you need ENV variables, make sure you export them manually")

# since we are testing comet, make sure it is set
assert os.environ["COMET_KEY"], "Please set COMET_KEY in your environment"

class myModel:

    def __init__(self):
        pass

    def predict(self):
        """
        Do something
        """
        return [1, 1, 1, 1]


apo_model = myModel()

# initialize with everything
cd = CoveoSessionRecList(
    model=apo_model,
    dataset=[1, 1, 1, 0],
    metadata={"categories": ["cat", "cat", "cat", "dog"]},
    logger=LOGGER.COMET,
    metadata_store= METADATA_STORE.LOCAL,
    **{
        "COMET_KEY":  os.environ["COMET_KEY"],
        "COMET_PROJECT_NAME": os.environ["COMET_PROJECT_NAME"],
        "COMET_WORKSPACE": os.environ["COMET_WORKSPACE"]
    }
)

# run the reclist
cd(verbose=True)

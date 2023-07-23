import json
import os
import time
from abc import ABC
from pathlib import Path
from functools import wraps
import pandas as pd
from reclist.charts import CHART_TYPE
from reclist.logs import LOGGER, logger_factory
from reclist.metadata import METADATA_STORE, metadata_store_factory
import datetime

def rec_test(test_type: str, display_type: CHART_TYPE = None):
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
        w.display_type = display_type
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
            model_name: str,
            logger: LOGGER = LOGGER.LOCAL,
            metadata_store: METADATA_STORE = METADATA_STORE.LOCAL,
            **kwargs,
            ):
        """
        :param model:
        :param dataset:
        """

        self.name = self.__class__.__name__
        self.model_name = model_name
        self._rec_tests = self.get_tests()
        self._test_results = []
        self.logger = logger
        self.logger_service = logger_factory(logger)(**kwargs)
        # if s3 is used, we need to specify the bucket
        self.metadata_bucket = kwargs["bucket"] if "bucket" in kwargs else None
        assert self.metadata_bucket is not None if metadata_store == METADATA_STORE.S3 else True, \
            "If using S3, you need to specify the bucket"
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
            self.model_name,
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
            # rich needs strings to display
            printable_result = None
            if isinstance(result['result'], float):
                printable_result = str(round(result['result'], 4))
            elif isinstance(result['result'], dict):
                printable_result = json.dumps(result['result'], indent=4)
            elif isinstance(result['result'], list):
                printable_result = json.dumps(
                    result['result'][:3] + ["..."],
                    indent=4
                )
            else:
                printable_result = str(result['result'])
            table.add_row(
                result['name'],
                result['description'],
                printable_result
                )
        # print out the table
        console = Console()
        console.print(table)

        return

    def __call__(self, verbose=True, *args, **kwargs):
        from rich.progress import track

        self.meta_store_path = self.create_data_store()
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
                    "display_type": str(test.display_type),
                }
            )
            self.logger_service.write(test.test_type, test_result)
        # finally, display all results in a table
        self._display_rich_table(self.name, self._test_results)
        # at the end, dump results to json and generate plots
        test_2_fig = self._generate_report_and_plot(self._test_results, self.meta_store_path)
        for test, fig in test_2_fig.items():
            self.logger_service.save_plot(name=test, fig=fig)

        return

    def _generate_report_and_plot(self, test_results: list, meta_store_path: str):
        """
        Store a copy of the results into a file in the metadata store

        TODO: decide what to do with artifacts
        """
        # dump results to json
        report_file_name = self._dump_results_to_json(test_results, meta_store_path)
        # generate and save plots if applicable
        test_2_fig = self._generate_plots(test_results)
        for test_name, fig in test_2_fig.items():
            if self.metadata_store == METADATA_STORE.LOCAL:
                # TODO: decide if we want to save the plot in S3 or not
                fig.savefig(os.path.join(meta_store_path, "plots", "{}.png".format(test_name)))
        # TODO: decide how store artifacts / if / where
        # self.store_artifacts(report_path)
        return test_2_fig

    def _generate_plots(self, test_results: list):
        test_2_fig = {}
        for test_result in test_results:
            display_type = test_result['display_type']
            fig = None
            if display_type == str(CHART_TYPE.SCALAR):
                # TODO: decide how to plot scalars
                pass
            elif display_type == str(CHART_TYPE.BARS):
                fig = self._bar_chart(test_result)
            elif display_type == str(CHART_TYPE.BINS):
                fig = self._bin_chart(test_result)
            # append fig to the mapping
            if fig is not None:
                test_2_fig[test_result['name']] = fig

        return test_2_fig

    def _bin_chart(self, test_result: dict):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(test_result['name'])
        data = test_result['result']
        assert isinstance(data, list), "data must be a list"
        ax.hist(data, color='lightgreen', ec='black')

        return fig

    def _bar_chart(self, test_result: dict):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title(test_result['name'])
        data = test_result['result'].keys()
        # cast keys to string; matplotlib requirement
        ax.bar([str(_) for _ in data], [test_result['result'][_] for _ in data])

        return fig


    def _dump_results_to_json(self, test_results: list, report_path: str):
        report = {
            "metadata": {
                "finish_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": self.model_name,
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






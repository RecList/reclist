import os
from enum import Enum
from abc import ABC, abstractmethod


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

    @abstractmethod
    def save_plot(self, name, fig, *args, **kwargs):
        pass


class LocalLogger(RecLogger):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        return

    def write(self, label, value):
        from rich import print
        print(f"[italic red]{label}[/italic red]:{value}")

        return

    def save_plot(self, name, fig, *args, **kwargs):
        pass


class NeptuneLogger(RecLogger):

    def __init__(self, *args, **kwargs):
        """

        In order of priority, use first the kwargs, then the env variables

        """
        super().__init__(*args, **kwargs)
        import neptune

        api_key = kwargs.get("NEPTUNE_KEY", os.environ["NEPTUNE_KEY"])
        project_name = kwargs.get("NEPTUNE_PROJECT_NAME", os.environ["NEPTUNE_PROJECT_NAME"])

        self.experiment = neptune.init_run(
            project=project_name,
            api_token=api_key,
        )

    def write(self, label, value):
        if isinstance(value, float):
            self.experiment[label] = value

    def save_plot(self, name, fig, *args, **kwargs):
        import tempfile
        with tempfile.NamedTemporaryFile() as temp:
            file_name = temp.name + ".png"
            fig.savefig(file_name)
            self.experiment[name].upload(file_name)


class CometLogger(RecLogger):

    def __init__(self, *args, **kwargs):
        """

        In order of priority, use first the kwargs, then the env variables

        """
        super().__init__(*args, **kwargs)
        from comet_ml import Experiment

        api_key = kwargs.get("COMET_KEY", os.environ["COMET_KEY"])
        project_name = kwargs.get("COMET_PROJECT_NAME", os.environ["COMET_PROJECT_NAME"])
        workspace = kwargs.get("COMET_WORKSPACE", os.environ["COMET_WORKSPACE"])

        # set up the experiment
        self.experiment = Experiment(
            api_key=api_key,
            project_name=project_name,
            workspace=workspace,
        )

    def write(self, label, value):
        if isinstance(value, float):
            self.experiment.log_metric(label, value)

    def save_plot(self, name, fig, *args, **kwargs):
        import tempfile
        with tempfile.NamedTemporaryFile() as temp:
            file_name = temp.name + ".png"
            fig.savefig(file_name)
            self.experiment.log_image(file_name, name=name)


def logger_factory(label) -> RecLogger:
    if label == LOGGER.COMET:
        return CometLogger
    elif label == LOGGER.NEPTUNE:
        return NeptuneLogger
    else:
        return LocalLogger

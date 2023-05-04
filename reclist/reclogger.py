from comet_ml import Experiment
import os
import logging
from enum import Enum

class LOGGER(Enum):
    LOCAL = 1
    COMET = 2


def logger_factory(label):
    if label == LOGGER.COMET:
        return CometLogger
    else:
        return RecLogger


class RecLogger:

    def __init__(self):
        pass

    def write(self, label, value):
        logging.info(f"{label}:{value}")


class CometLogger(RecLogger):

    def __init__(self):
        super().__init__()
        self.experiment = Experiment(
            api_key=os.environ["COMET_KEY"],
            project_name=os.environ["COMET_PROJECT_NAME"],
            workspace=os.environ["COMET_WORKSPACE"],
        )

    def write(self, label, value):
        self.experiment.log_metric(label, value)


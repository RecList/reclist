from comet_ml import Experiment


def logger_factory(label):
    if label == "comet":
        return CometLogger
    else:
        return RecLogger



class RecLogger:

    def __init__(self):
        pass

    def write(self, label, value):
        pass


class CometLogger(RecLogger):

    def __init__(self):
        self.experiment = Experiment(
            api_key="",
            project_name="",
            workspace="",
        )

    def write(self, label, value):
        self.experiment.log_metric(label, value)


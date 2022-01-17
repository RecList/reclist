from reclist.abstractions import RecModel


class BBCSoundsLightFMSimulatorModel(RecModel):
    """
    LightFM implementation for BBC Sounds Dataset
    The model is not trained here, only predictions are being loaded directly from the specified GCP bucket.
    """
    model_name = "lightfm"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict(self, prediction_input, *args, **kwargs):
        """
        Predicts the top 10 similar resource IDs recommended for each user according
        to the resource IDs that they've watched

        :param prediction_input: a list of lists containing a dictionary for
                                 each resource ID watched by that user
        :return:
        """
        all_predictions = []

        return all_predictions

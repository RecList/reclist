"""

Example script to run a RecList over a dummy, free-form dataset.

Note that if you use LOGGER.COMET, you should uncomment the COMET variables to get the script to
log in your Comet account the relevant metrics as they are computed by the rec tests. Of course,
make sure the Comet python library is installed in your environment.

"""

import numpy as np
from reclist.logs import LOGGER
from reclist.similarity_models import FakeSimilarityModel
from reclist.metadata import METADATA_STORE
from reclist.reclist import RecList
from random import randint, choice
import os
from reclist.reclist import rec_test
from reclist.charts import CHART_TYPE

class FreeSessionRecList(RecList):

    def __init__(
        self,
        dataset,
        metadata,
        predictions,
        model_name,
        logger: LOGGER,
        metadata_store: METADATA_STORE,
        **kwargs
    ):
        super().__init__(
            model_name,
            logger,
            metadata_store,
            **kwargs
        )
        self.dataset = dataset
        self.metadata = metadata
        self.predictions = predictions
        self.similarity_model = kwargs.get("similarity_model", None)

        return

    @rec_test(test_type="LessWrong", display_type=CHART_TYPE.SCALAR)
    def less_wrong(self):
        truths = self.dataset
        predictions = self.predictions
        model_misses = [(t, p) for t, p in zip(truths, predictions) if t != p]
        similarity_scores = [
            self.similarity_model.similarity_gradient(t, p) for t, p in model_misses
        ]

        return np.average(similarity_scores)

    @rec_test(test_type="SlicedAccuracy", display_type=CHART_TYPE.SCALAR)
    def sliced_accuracy(self):
        """
        Compute the accuracy by slice
        """
        from reclist.metrics.standard_metrics import accuracy_per_slice

        return accuracy_per_slice(
            self.dataset, self.predictions, self.metadata["categories"]
        )

    @rec_test(test_type="Accuracy", display_type=CHART_TYPE.SCALAR)
    def accuracy(self):
        """
        Compute the accuracy
        """
        from sklearn.metrics import accuracy_score

        return accuracy_score(self.dataset, self.predictions)

    @rec_test(test_type="AccuracyByCountry", display_type=CHART_TYPE.BARS)
    def accuracy_by_country(self):
        """
        Compute the accuracy by country
        """
        # TODO: note that is a static test, used to showcase the bin display
        from random import randint
        return {"US": randint(0, 100), "CA": randint(0, 100), "FR": randint(0, 100) }


try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    print("Dotenv not loaded: if you need ENV variables, make sure you export them manually")


class DummyModel:

    def __init__(self, n):
        self.n = n

    def predict(self):
        """
        Do something
        """
        from random import randint
        return [randint(0, 1) for _ in range(self.n)]


# create a dataset randomly

n = 10000
apo_model = DummyModel(n)
dataset = [randint(0, 1) for _ in range(n)]
metadata = {"categories": [choice(["cat", "dog", "capybara"]) for _ in range(n)]}
predictions = apo_model.predict()
my_sim_model = FakeSimilarityModel()
assert len(dataset) == len(predictions), "dataset, predictions must have the same length"


# initialize with everything
cd = FreeSessionRecList(
    dataset=dataset,
    metadata=metadata,
    predictions=predictions,
    model_name="myRandomModel",
    logger=LOGGER.LOCAL,
    metadata_store= METADATA_STORE.LOCAL,
    similarity_model=my_sim_model,
    # bucket=os.environ["S3_BUCKET"], # if METADATA_STORE.LOCAL you don't need this!
    # COMET_KEY=os.environ["COMET_KEY"], # if LOGGER.COMET, make sure you have the env
    # COMET_PROJECT_NAME=os.environ["COMET_PROJECT_NAME"], # if LOGGER.COMET, make sure you have the env
    # COMET_WORKSPACE=os.environ["COMET_WORKSPACE"], # if LOGGER.COMET, make sure you have the env
)

# run reclist
cd(verbose=True)


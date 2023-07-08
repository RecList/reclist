from reclist.logs import LOGGER
from reclist.similarity_models import SkigramSimilarityModel
from reclist.reclist import FreeSessionRecList
from reclist.metadata import METADATA_STORE
import pandas as pd
from reclist.reclist import DFSessionRecList
import numpy as np
import os

try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    print("Dotenv not loaded: if you need ENV variables, make sure you export them manually")


class myModel:

    def __init__(self, n):
        self.n = n

    def predict(self):
        """
        Do something
        """
        from random import randint
        return [randint(0, 1) for _ in range(self.n)]


# create a dataset randomly
from random import randint, choice
n = 10000
apo_model = myModel(n)
dataset = [randint(0, 1) for _ in range(n)]
metadata = {"categories": [choice(["cat", "dog", "capybara"]) for _ in range(n)]}
predictions = apo_model.predict()
my_sim_model = SkigramSimilarityModel()
assert len(dataset) == len(predictions), "dataset, predictions must have the same length"


# initialize with everything
cd = FreeSessionRecList(
    dataset=dataset,
    metadata=metadata,
    predictions=predictions,
    model_name="myRandomModel",
    logger=LOGGER.COMET,
    metadata_store= METADATA_STORE.LOCAL,
    similarity_model=my_sim_model,
    bucket=os.environ["S3_BUCKET"],
    COMET_KEY=os.environ["COMET_KEY"],
    COMET_PROJECT_NAME=os.environ["COMET_PROJECT_NAME"],
    COMET_WORKSPACE=os.environ["COMET_WORKSPACE"],
)

# run reclist
cd(verbose=True)

# test the similarity model with open ai :clownface:
#sim_model = GPT3SimilarityModel(api_key=os.environ["OPENAI_API_KEY"])
#p1 = {
#    "name": "logo-print cotton cap",
#    "brand": 'Palm Angels',
#    "description": '''
#    Known for a laid-back aesthetic, Palm Angels knows how to portray its Californian inspiration. This classic cap carries the brand's logo printed on the front, adding a touch of recognition to a relaxed look.
#    '''
#}
#p2 = {
#    "name": "monogram badge cap",
#    "brand": 'Balmain',
#    "description": '''
#    Blue cotton monogram badge cap from Balmain featuring logo patch to the front, mesh detailing, fabric-covered button at the crown and adjustable fit.
#    '''
#}
#similarity_judgement = sim_model.similarity_binary(p1, p2, verbose=False)
#print("P1 {} and P2 {} are similar: {}".format(p1["name"], p2["name"], similarity_judgement))



"""

Example script to run a RecList over the EvalRS2023 dataset.

The dataset files should be places in the folder evalrs_dataset_KDD23.

Download the dataset before running the script:

https://github.com/RecList/evalRS-KDD-2023/blob/main/evaluation/utils.py


Note that if you use LOGGER.NEPTUNE, you should uncomment the NEPTUNE variables to get the script to
log in your Neptune account the relevant metrics as they are computed by the rec tests. Of course,
make sure the Neptune python library is installed in your environment.

"""

from reclist.logs import LOGGER
from reclist.metadata import METADATA_STORE
import pandas as pd
import numpy as np
import os
from reclist.reclist import rec_test
from reclist.reclist import RecList
from random import choice

class DFSessionRecList(RecList):

    def __init__(
        self,
        dataset,
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
        self._y_preds = predictions
        self._y_test = kwargs.get("y_test", None)
        self.similarity_model = kwargs.get("similarity_model", None)

        return

    @rec_test('HIT_RATE')
    def hit_rate_at_100(self):
        hr = self.hit_rate_at_k(self._y_preds, self._y_test, k=100)
        return hr

    def hit_rate_at_k(self, y_pred: pd.DataFrame, y_test: pd.DataFrame, k: int):
        """
        N = number test cases
        M = number ground truth per test case
        """
        hits = self.hits_at_k(y_pred, y_test, k)  # N x M x k
        hits = hits.max(axis=1)  # N x k
        return hits.max(axis=1).mean()  # 1

    def hits_at_k(self, y_pred: pd.DataFrame, y_test: pd.DataFrame, k: int):
        """
        N = number test cases
        M = number ground truth per test case
        """
        y_test_mask = y_test.values != -1  # N x M

        y_pred_mask = y_pred.values[:, :k] != -1  # N x k

        y_test = y_test.values[:, :, None]  # N x M x 1
        y_pred = y_pred.values[:, None, :k]  # N x 1 x k

        hits = y_test == y_pred  # N x M x k
        hits = hits * y_test_mask[:, :, None]  # N x M x k
        hits = hits * y_pred_mask[:, None, :]  # N x M x k

        return hits


class EvalRSSimpleModel:
    """
    This is a dummy model that returns random predictions on the EvalRS dataset.
    """
    def __init__(self, items: pd.DataFrame, top_k: int=10, **kwargs):
        self.items = items
        self.top_k = top_k
        print("Received additional arguments: {}".format(kwargs))
        return

    def predict(self, user_ids: pd.DataFrame) -> pd.DataFrame:
        k = self.top_k
        num_users = len(user_ids)
        pred = self.items.sample(n=k*num_users, replace=True).index.values
        pred = pred.reshape(num_users, k)
        pred = np.concatenate((user_ids[['user_id']].values, pred), axis=1)
        return pd.DataFrame(pred, columns=['user_id', *[str(i) for i in range(k)]]).set_index('user_id')


"""
    We now load the dataset
"""

if __name__ == '__main__':

    print("\n\n ======> Loading the EvalRS2023 dataset from the local folder\n\n")
    # make sure files are there
    assert os.path.isfile('evalrs_dataset_KDD23/evalrs_events.csv'), "Please download the dataset first!"

    df_events = pd.read_csv('evalrs_dataset_KDD23/evalrs_events.csv', index_col=0, dtype='int32')
    df_tracks = pd.read_csv('evalrs_dataset_KDD23/evalrs_tracks.csv',
                                    dtype={
                                        'track_id': 'int32',
                                        'artist_id': 'int32'
                                    }).set_index('track_id')

    df_users = pd.read_csv('evalrs_dataset_KDD23/evalrs_users.csv',
                                dtype={
                                    'user_id': 'int32',
                                    'playcount': 'int32',
                                    'country_id': 'int32',
                                    'timestamp': 'int32',
                                    'age': 'int32',
                                })

    """
        Here we would normally train a model, but we just return random predictions.
    """
    my_df_model = EvalRSSimpleModel(df_tracks, top_k=10)
    df_predictions = my_df_model.predict(df_users)
    # build a mock dataset for the golden standard
    all_tracks = df_tracks.index.values

    df_dataset = pd.DataFrame(
        {
            'track_id': [choice(all_tracks) for _ in range(len(df_predictions))]
        }
    )

    """
        Here we use RecList to run the evaluation.
    """

    # initialize with everything
    cdf = DFSessionRecList(
        dataset=df_events,
        model_name="myDataFrameRandomModel",
        predictions=df_predictions,
        # I can specify the gold standard here, or doing it in the init of course
        y_test=df_dataset,
        logger=LOGGER.LOCAL,
        metadata_store=METADATA_STORE.LOCAL,
        # bucket=os.environ["S3_BUCKET"], # if METADATA_STORE.LOCAL you don't need this!
        #NEPTUNE_KEY=os.environ["NEPTUNE_KEY"],
        #NEPTUNE_PROJECT_NAME=os.environ["NEPTUNE_PROJECT_NAME"],
    )

    # run reclist
    cdf(verbose=True)

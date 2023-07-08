from reclist.logs import LOGGER
from reclist.metadata import METADATA_STORE
import pandas as pd
from reclist.reclist import DFSessionRecList
import numpy as np
import os
from random import choice

class MySuperModel:
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


print("\n\n ======> Loading dataset. \n\n")
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

my_df_model = MySuperModel(df_tracks, top_k=10)
df_predictions = my_df_model.predict(df_users)
# build a mock dataset for the golden standard
all_tracks = df_tracks.index.values

df_dataset = pd.DataFrame(
    {
        'track_id': [choice(all_tracks) for _ in range(len(df_predictions))]
    }
)

# initialize with everything
cdf = DFSessionRecList(
    dataset=df_events,
    model_name="myDataFrameRandomModel",
    predictions=df_predictions,
    # I can specify the gold standard here, or doing it in the init of course
    y_test=df_dataset,
    logger=LOGGER.NEPTUNE,
    metadata_store=METADATA_STORE.LOCAL,
    bucket=os.environ["S3_BUCKET"],
    NEPTUNE_KEY=os.environ["NEPTUNE_KEY"],
    NEPTUNE_PROJECT_NAME=os.environ["NEPTUNE_PROJECT_NAME"],
)

# run reclist
cdf(verbose=True)

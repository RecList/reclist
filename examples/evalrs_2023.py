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
from reclist.reclist import RecList, CHART_TYPE
from random import choice
from gensim.models import KeyedVectors


class DFSessionRecList(RecList):

    """
    This is a simple RecList over the EvalRS2023 dataset.

    Once you are familiar with the RecList abstractions, you can head over to the EvalRS Challenge
    and explore a full-fledge suite of tests over the same dataset:

    https://github.com/RecList/evalRS-KDD-2023/blob/main/evaluation/EvalRSReclist.py
    """

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
        self._user_metadata = kwargs.get("user_metadata", None)
        if not isinstance(self._user_metadata, type(None)):
            self._user_metadata = self._user_metadata.set_index("user_id")
        self.similarity_model = kwargs.get("similarity_model", None)


    @rec_test(test_type='HIT_RATE')
    def hit_rate_at_100(self):
        from reclist.metrics.standard_metrics import hit_rate_at_k
        hr = hit_rate_at_k(self._y_preds, self._y_test, k=100)
        return hr

    @rec_test(test_type='MRR')
    def mrr_at_100(self):
        from reclist.metrics.standard_metrics import mrr_at_k

        return mrr_at_k(self._y_preds, self._y_test, k=100)

    @rec_test(test_type='MRED_COUNTRY', display_type=CHART_TYPE.BARS)
    def mred_country(self):
        country_list = ["US", "RU", "DE", "UK", "PL", "BR", "FI", "NL", "ES", "SE", "UA", "CA", "FR", "NaN"]
        
        user_countries = self._user_metadata.loc[self._y_test.index, ['country']].fillna('NaN')
        valid_country_mask = user_countries['country'].isin(country_list)
        y_pred_valid = self._y_preds[valid_country_mask]
        y_test_valid = self._y_test[valid_country_mask]
        user_countries = user_countries[valid_country_mask]

        return self.miss_rate_equality_difference(y_pred_valid, y_test_valid, user_countries, 'country')

    @rec_test(test_type='BEING_LESS_WRONG')
    def being_less_wrong(self):
        from reclist.metrics.standard_metrics import hits_at_k

        hits = hits_at_k(self._y_preds, self._y_test, k=100).max(axis=2)
        misses = (hits == False)
        miss_gt_vectors = self.similarity_model[self._y_test.loc[misses, 'track_id'].values.reshape(-1)]
        # we calculate the score w.r.t to the first prediction
        miss_pred_vectors = self.similarity_model[self._y_preds.loc[misses, '0'].values.reshape(-1)]

        return float(self.cosine_sim(miss_gt_vectors, miss_pred_vectors).mean())
    
    def cosine_sim(self, u: np.array, v: np.array) -> np.array:
        return np.sum(u * v, axis=-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1))

    def miss_rate_at_k_slice(self,
                                   y_preds: pd.DataFrame,
                                   y_test: pd.DataFrame,
                                   slice_info: pd.DataFrame,
                                   slice_key: str):
        from reclist.metrics.standard_metrics import misses_at_k
        # get false positives
        m = misses_at_k(y_preds, y_test, k=100).min(axis=2)
        # convert to dataframe
        m = pd.DataFrame(m, columns=['mr'], index=y_test.index)
        # grab slice info
        m[slice_key] = slice_info[slice_key].values
        # group-by slice and get per-slice mrr
        return m.groupby(slice_key)['mr'].agg('mean')

    def miss_rate_equality_difference(self,
                                      y_preds: pd.DataFrame,
                                      y_test: pd.DataFrame,
                                      slice_info: pd.DataFrame,
                                      slice_key: str):
        from reclist.metrics.standard_metrics import misses_at_k

        mr_per_slice = self.miss_rate_at_k_slice(y_preds, y_test, slice_info, slice_key)
        mr = misses_at_k(y_preds, y_test, k=100).min(axis=2).mean()
        # take negation so that higher values => better fairness
        mred = -(mr_per_slice-mr).abs().mean()
        res = mr_per_slice.to_dict()
        return {'mred': mred, 'mr': mr, **res}
    

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


if __name__ == '__main__':

    print("\n\n ======> Loading the EvalRS2023 dataset from the local folder\n\n")
    # make sure files are there
    assert os.path.isfile('evalrs_dataset_KDD_2023/evalrs_events.parquet'), "Please download the dataset first!"

    df_events = pd.read_parquet('evalrs_dataset_KDD_2023/evalrs_events.parquet')
    df_tracks = pd.read_parquet('evalrs_dataset_KDD_2023/evalrs_tracks.parquet').set_index('track_id')
    df_users = pd.read_parquet('evalrs_dataset_KDD_2023/evalrs_users.parquet')

    similarity_model = KeyedVectors.load('evalrs_dataset_KDD_2023/song2vec.wv')

    """
        Here we would normally train a model, but we just return random predictions.
    """
    my_df_model = EvalRSSimpleModel(df_tracks, top_k=10)
    df_predictions = my_df_model.predict(df_users[['user_id']])
    # build a mock dataset for the golden standard
    all_tracks = df_tracks.index.values
    df_dataset = pd.DataFrame(
        {
            'user_id': df_predictions.index.tolist(),
            'track_id': [choice(all_tracks) for _ in range(len(df_predictions))]
        }
    ).set_index('user_id')


    """
        Here we use RecList to run the evaluation.
    """
    # initialize with everything
    cdf = DFSessionRecList(
        dataset=df_events,
        model_name="myDataFrameRandomModel",
        predictions=df_predictions,
        # you can specify the gold standard here, or doing it in the init
        y_test=df_dataset,
        logger=LOGGER.LOCAL,
        metadata_store=METADATA_STORE.LOCAL,
        similarity_model=similarity_model,
        user_metadata=df_users,
    )
    # run reclist
    cdf(verbose=True)

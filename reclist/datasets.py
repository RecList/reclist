import json
import os
import tempfile
import zipfile

import pandas as pd

from reclist.abstractions import RecDataset
from reclist.utils.config import *
from sklearn.model_selection import train_test_split
import numpy as np


class SyntheticDataset(RecDataset):
    """
    Synthetic Dataset
    -------------------

    This dataset is used for testing purposes. It generates a random dataset with the following parameters:

    - n_users: number of users
    - n_items: number of items
    - n_interactions: number of interactions
    - size: size of the dataset
    - seed: seed for the random number generator
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_users = None
        self.n_items = None
        self.n_interactions=None, 
        self.size=100, 
        self.seed=42

    def set_n_users(self, n_users):
        self.n_users = n_users

    
    def produce_dataset(self):
        if self.n_interactions is None:
            self.n_interactions = 1 
        user_ids = np.random.randint(0, self.n_users, self.size)
        item_ids = np.random.randint(0, self.n_items, self.size)
        interactions = np.random.randint(0, self.n_interactions, self.size)
        return pd.DataFrame({'user_id':user_ids, 'item_id':item_ids, 'interactions':interactions})
    
    def get_train_test(self, ratio=0.2):
        self.n_users = 10
        self.n_items = 20
        self.size = 100
        self.n_interactions = 5
        self.seed = 42
        self.ratio = 0.2
        data = self.produce_dataset()
        return train_test_split(data, test_size=ratio, random_state=self.seed)
    
    def make_user_data(self, user_id, feature_type='categorical', n_features=10):
       pass

    def make_item_data(self, item_id, feature_type={'categorical':8, 'regression':2}, n_features=10):
        pass
    
    def load(self, **kwargs):
        print("Loading Synthetic Dataset ...")
        train, test = self.get_train_test()
        features = ['user_id', 'item_id']
        to_predict = ['interactions']
        self._x_train = train[features]
        self._y_train = train[to_predict]
        self._x_test = train[features]
        self._y_test = test[to_predict]
        self._catalog = None
        self._x_test = train[to_predict]
    


class MovieLensDatasetDF(RecDataset):
    """
    MovieLens 25M Dataset

    Reference: https://files.grouplens.org/datasets/movielens/ml-25m-README.html
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        print("MovieLensDatasetDF")

    def load(self, **kwargs):
        cache_dir = get_cache_directory()
        filepath = os.path.join(cache_dir, "movielens_25m_useritem.zip")

        if not os.path.exists(filepath) or self.force_download:
            download_with_progress(MOVIELENS_UI_DATASET_S3_URL, filepath)

        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(filepath, "r") as zip_file:
                zip_file.extractall(temp_dir)
            self._x_train = pd.read_parquet(
                os.path.join(temp_dir, "movielens_reclist", "movielens_x_train.pk")
            )
            self._y_train = pd.read_parquet(
                os.path.join(temp_dir, "movielens_reclist", "movielens_y_train.pk")
            )
            self._x_test = pd.read_parquet(
                os.path.join(temp_dir, "movielens_reclist", "movielens_x_test.pk")
            )
            self._y_test = pd.read_parquet(
                os.path.join(temp_dir, "movielens_reclist", "movielens_y_test.pk")
            )
            self._catalog = pd.read_parquet(
                os.path.join(temp_dir, "movielens_reclist", "movielens_catalog.pk")
            )


class MovieLensDataset(RecDataset):
    """
    MovieLens 25M Dataset

    Reference: https://files.grouplens.org/datasets/movielens/ml-25m-README.html
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self):
        cache_dir = get_cache_directory()
        filepath = os.path.join(cache_dir, "movielens_25m.zip")

        if not os.path.exists(filepath) or self.force_download:
            download_with_progress(MOVIELENS_DATASET_S3_URL, filepath)

        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(filepath, "r") as zip_file:
                zip_file.extractall(temp_dir)
            with open(os.path.join(temp_dir, "dataset.json")) as f:
                data = json.load(f)

        self._x_train = data["x_train"]
        self._y_train = None
        self._x_test = data["x_test"]
        self._y_test = data["y_test"]
        self._catalog = self._convert_catalog_keys(data["catalog"])

    def _convert_catalog_keys(self, catalog):
        """
        Convert catalog keys from string to integer type

        JSON encodes all keys to strings, so the catalog dictionary
        will be loaded up string representation of movie IDs.
        """
        converted_catalog = {}
        for k, v in catalog.items():
            converted_catalog[int(k)] = v
        return converted_catalog


class CoveoDataset(RecDataset):
    """
    Coveo SIGIR data challenge dataset
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self):
        cache_directory = get_cache_directory()
        filename = os.path.join(
            cache_directory, "coveo_sigir.zip"
        )  # TODO: make var somewhere

        if not os.path.exists(filename) or self.force_download:
            download_with_progress(COVEO_INTERACTION_DATASET_S3_URL, filename)

        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
            with open(os.path.join(temp_dir, "dataset.json")) as f:
                data = json.load(f)

        self._x_train = data["x_train"]
        self._y_train = None
        self._x_test = data["x_test"]
        self._y_test = data["y_test"]
        self._catalog = data["catalog"]


class SpotifyDataset(RecDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load(self):
        data = self.load_spotify_playlist_dataset()
        self._x_train = data["train"]
        self._y_train = None
        self._x_test = data["test"]
        self._y_test = None
        self._catalog = data["catalog"]

        # generate NEP dataset here for now
        test_pairs = [
            (playlist[:-1], [playlist[-1]])
            for playlist in self._x_test
            if len(playlist) > 1
        ]
        self._x_test, self._y_test = zip(*test_pairs)

    def load_spotify_playlist_dataset(self):

        cache_directory = get_cache_directory()
        filename = os.path.join(
            cache_directory, "small_spotify_playlist.zip"
        )  # TODO: make var somewhere

        if not os.path.exists(filename) or self.force_download:
            download_with_progress(SPOTIFY_PLAYLIST_DATASET_S3_URL, filename)

        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(filename, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
            with open(os.path.join(temp_dir, "dataset.json")) as f:
                data = json.load(f)
        return data

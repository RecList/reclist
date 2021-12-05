import json
import tempfile
import zipfile
import os
from reclist.abstractions import RecDataset
from reclist.utils.config import *


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
        filename = os.path.join(cache_directory, "coveo_sigir.zip")  # TODO: make var somewhere

        if not os.path.exists(filename) or self.force_download:
            download_with_progress(COVEO_INTERACTION_DATASET_S3_URL, filename)

        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            with open(os.path.join(temp_dir, 'dataset.json')) as f:
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
        self._x_test = data['test']
        self._y_test = None
        self._catalog = data["catalog"]

        # generate NEP dataset here for now
        test_pairs = [(playlist[:-1], [playlist[-1]])  for playlist in self._x_test if len(playlist) > 1]
        self._x_test, self._y_test = zip(*test_pairs)

    def load_spotify_playlist_dataset(self):

        cache_directory = get_cache_directory()
        filename = os.path.join(cache_directory, "small_spotify_playlist.zip")   # TODO: make var somewhere

        if not os.path.exists(filename) or self.force_download:
            download_with_progress(SPOTIFY_PLAYLIST_DATASET_S3_URL, filename)

        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            with open(os.path.join(temp_dir, 'dataset.json')) as f:
                data = json.load(f)
        return data

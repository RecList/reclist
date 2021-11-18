import json
import tempfile
import zipfile
import random
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
        self._catalog = data["catalog"]


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

    def __init__(self, k: int = 5, **kwargs):
        self.k = k
        super().__init__(**kwargs)

    def load(self):
        data = self.load_spotify_playlist_dataset()
        x_test, y_test = self.preprocess_spotify_playlist_data()
        self._x_train = data["x_train"]
        self._y_train = None
        self._x_test = x_test
        self._y_test = y_test
        self._catalog = data["metadata"]

    def load_spotify_playlist_dataset(self):

        cache_directory = get_cache_directory()
        filename = os.path.join(cache_directory, "spotify_playlist.zip")   # TODO: make var somewhere

        if not os.path.exists(filename) or self.force_download:
            download_with_progress(SPOTIFY_PLAYLIST_DATASET_S3_URL, filename)

        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            with open(os.path.join(temp_dir, 'dataset.json')) as f:
                data = json.load(f)
        return data

    def preprocess_spotify_playlist_data(
        self,
        shuffle: bool = False,
        seed: int = 0
    ):
        x_test = []
        y_test = []
        for playlist in self.data['test']:
            if len(playlist['tracks']) <= self.k:
                continue
            if shuffle:
                random.seed(seed)
                all_idx = list(range(len(playlist['tracks'])))
                seeded_idx = random.sample(all_idx, k=self.k)
                held_out_idx = set(all_idx) - set(seeded_idx)
                seeded_tracks = [playlist['tracks'][idx] for idx in seeded_idx]
                held_out_tracks = [playlist['tracks'][idx]
                                   for idx in held_out_idx]
                assert len(seeded_tracks) + \
                    len(held_out_tracks) == len(playlist['tracks'])
            else:
                seeded_tracks = playlist['tracks'][:self.k]
                held_out_tracks = playlist['tracks'][self.k:]
            x_test.append(seeded_tracks)
            y_test.append(held_out_tracks)

        return x_test, y_test

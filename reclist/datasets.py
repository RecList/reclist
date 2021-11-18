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

    def __init__(self):
        x_train, y_train, x_test, y_test, catalog = self._load_movielens_dataset()
        super().__init__(x_train, y_train, x_test, y_test, catalog)

    def _load_movielens_dataset(self):
        cache_dir = get_cache_directory()
        filepath = os.path.join(cache_dir, "movielens_25m.zip")

        if not os.path.exists(filepath):
            download_with_progress(MOVIELENS_DATASET_S3_URL, filepath)

        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(filepath, "r") as zip_file:
                zip_file.extractall(temp_dir)
            with open(os.path.join(temp_dir, "dataset.json")) as f:
                data = json.load(f)
        return data["x_train"], None, data["x_test"], data["y_test"], data["catalog"]


class CoveoDataset(RecDataset):
    """
    Coveo SIGIR data challenge dataset
    """
    def __init__(self):
        x_train, y_train, x_test, y_test, catalog = self.load_coveo_interaction_dataset()
        super().__init__(x_train, y_train, x_test, y_test, catalog)

    def load_coveo_interaction_dataset(self):
        cache_directory = get_cache_directory()
        filename = os.path.join(cache_directory, "coveo_sigir.zip")  # TODO: make var somewhere

        if not os.path.exists(filename):
            download_with_progress(COVEO_INTERACTION_DATASET_S3_URL, filename)

        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            with open(os.path.join(temp_dir, 'dataset.json')) as f:
                data = json.load(f)

        return data['x_train'], None, data['x_test'], data['y_test'], data['catalog']


class SpotifyDataset(RecDataset):

    def __init__(self, k: int = 5):
        self.k = k
        self.load_spotify_playlist_dataset()
        x_test, y_test = self.preprocess_spotify_playlist_data()
        super().__init__(self.data['train'], None,
                         x_test, y_test, self.data['metadata'])

    def load_spotify_playlist_dataset(self):

        cache_directory = get_cache_directory()
        filename = os.path.join(cache_directory, "spotify_playlist.zip")   # TODO: make var somewhere

        if not os.path.exists(filename):
            download_with_progress(SPOTIFY_PLAYLIST_DATASET_S3_URL, filename)

        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(filename, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            # with gzip.open(filename, 'rb') as f_in:
            #     with open(temp_dir, 'wb') as f_out:
            #         shutil.copyfileobj(f_in, f_out)
            with open(os.path.join(temp_dir, 'dataset.json')) as f:
                self.data = json.load(f)

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

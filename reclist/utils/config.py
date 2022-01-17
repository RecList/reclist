import json
from appdirs import *
from pathlib import Path
import requests
from tqdm import tqdm
from enum import Enum
from google.cloud import storage

COVEO_INTERACTION_DATASET_S3_URL = 'https://reclist-datasets-6d3c836d-6djh887d.s3.us-west-2.amazonaws.com/coveo_sigir.zip'
SPOTIFY_PLAYLIST_DATASET_S3_URL = 'https://reclist-datasets-6d3c836d-6djh887d.s3.us-west-2.amazonaws.com/small_spotify_playlist.zip'
MOVIELENS_DATASET_S3_URL = "https://reclist-datasets-6d3c836d-6djh887d.s3.us-west-2.amazonaws.com/movielens_25m.zip"
BBC_SOUNDS_TRAIN_GCP_URL = "bbc-datalab-sounds-pesquet/dataset/sampled/train/train_1.ndjson"
BBC_SOUNDS_TEST_GCP_URL = "bbc-datalab-sounds-pesquet/dataset/sampled/test/test.ndjson"
BBC_SOUNDS_PREDICTIONS = "bbc-datalab-sounds-pesquet/predictions_data/xantus/2022-01-27_13:53:56.json"


def download_file(blob_path, destination, project="datalab-user-projects-be57", bucket="parrots-projects-data"):
    storage_client = storage.Client(project)
    bucket = storage_client.get_bucket(bucket)
    blob = bucket.blob(blob_path)
    blob.download_to_filename(destination)


def load_json_from_bucket(filename, project="datalab-user-projects-be57", bucket_name='parrots-projects-data'):
    """ Loads json from a GCS bucket """
    storage_client = storage.Client(project)
    bucket = storage_client.get_bucket(bucket_name)
    # get the blob
    blob = bucket.get_blob(filename)
    # load blob using json
    file_data = json.loads(blob.download_as_string())
    return file_data


def load_predictions(filename, bucket_name='parrots-projects-data'):
    json_object = load_json_from_bucket(filename, bucket_name)
    predictions = json_object['predictions']
    metadata = json_object['metadata']
    return predictions, metadata


def download_with_progress(url, destination):
    """
    Downloads a file with a progress bar

    :param url: url from which to download from
    :destination: file path for saving data
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise SystemExit(e)
    with tqdm.wrapattr(open(destination, "wb"), "write",
                       miniters=1, desc=url.split('/')[-1],
                       total=int(response.headers.get('content-length', 0))) as fout:
        for chunk in response.iter_content(chunk_size=4096):
            fout.write(chunk)


def get_cache_directory():
    """
    Returns the cache directory on the system
    """
    appname = "reclist"
    appauthor = "reclist"
    cache_dir = user_cache_dir(appname, appauthor)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    return cache_dir


class Dataset(Enum):
    COVEO = 'coveo'
    COVEO_INTERNAL = 'coveo-internal'
    MOVIELENS = 'movielens'
    SPOTIFY = 'spotify'

"""
    This is a sample script showing how to get started with reclist. In particular,
    we are targeting the use case of session-based recommendations.

    We train a simple, yet effective prod2vec baseline model (https://arxiv.org/abs/2007.14906),
    re-using for convenience a "training embedding" function already implemented by recsys. The script
    shows how easy it is to run behavioral tests on a target dataset, in this case a wrapper around a large
    music dataset (the Spotify Million Playlist dataset:
    https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge).

    If you want to run your own model on the same dataset and the same behavioral test suite, just implement
    your model prediction using the RecModel, and pass it to the RecList.
"""
from reclist.datasets import SpotifyDataset
from reclist.recommenders.prod2vec import SpotifyP2VRecModel
from reclist.reclist import SpotifySessionRecList

if __name__ == "__main__":

    # get the Spotify million playlist dataset as a RecDataset object
    spotify_dataset = SpotifyDataset()

    # re-use a skip-gram model from reclist to train a latent product space, to be used
    # (through knn) to build a recommender
    model = SpotifyP2VRecModel()
    model.train(spotify_dataset.x_train)

    # instantiate rec_list object, prepared with standard quantitative tests
    # and sensible behavioral tests (check the paper for details!)
    rec_list = SpotifySessionRecList(
        model=model,
        dataset=spotify_dataset
    )
    # invoke rec_list to run tests
    rec_list(verbose=True)

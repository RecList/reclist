"""
    This is a sample script showing how to get started with reclist. In particular,
    we are targeting a "similar items" use cases, such as for example a item recommender:
    if a shopper is browsing a product listing and clicked on item X, what other item
    might they be interested in that is similar in type to item X?

    We train a simple, yet effective prod2vec baseline model (https://arxiv.org/abs/2007.14906),
    re-using for convenience a "training embedding" function already implemented by recsys. The script
    show how easy it is to run behavioral tests on a target dataset, in this case a wrapper around a large
    dataset widely used for benchmarking recommendation systems (MovieLens 25M dataset: https://grouplens.org/datasets/movielens/)

    If you want to run your own model on the same dataset and the same behavioral test suite, just implement
    your model prediction using the RecModel, and pass it to the RecList.
"""
from reclist.datasets import MovieLensDataset
from reclist.reclist import MovieLensSimilarItemRecList
from reclist.recommenders.prod2vec import MovieLensP2VRecModel
from reclist.abstractions import RecModel, RecDataset, RecList

if __name__ == "__main__":

    # get the MovieLens 25M dataset as a RecDataset object
    movielens_dataset: RecDataset= MovieLensDataset()

    # re-use a skip-gram model from reclist to train a latent product space, to be used
    # (through knn) to build a recommender
    model: RecModel= MovieLensP2VRecModel()
    model.train(movielens_dataset.x_train)

    # instantiate rec_list object, prepared with standard quantitative tests
    # and sensible behavioral tests (check the paper for details!)
    rec_list: RecList = MovieLensSimilarItemRecList(model=model, dataset=movielens_dataset)

    # invoke rec_list to run tests
    rec_list(verbose=True)

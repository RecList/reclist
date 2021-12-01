#!/usr/bin/env python

"""Tests for `reclist` package."""

from reclist.datasets import *
from reclist.metrics.standard_metrics import mrr_at_k
from reclist.datasets import CoveoDataset, MovieLensDataset
from reclist.recommenders.prod2vec import CoveoP2VRecModel, MovieLensP2VRecModel
from reclist.reclist import CoveoCartRecList, MovieLensSimilarItemRecList

def test_basic_dataset_downloading():
    CoveoDataset()
    MovieLensDataset()


def test_examples():
    # get the coveo data challenge dataset as a RecDataset object
    coveo_dataset = CoveoDataset()

    # re-use a skip-gram model from reclist to train a latent product space, to be used
    # (through knn) to build a recommender
    model = CoveoP2VRecModel()
    model.train(coveo_dataset.x_train)

    # instantiate rec_list object, prepared with standard quantitative tests
    # and sensible behavioral tests (check the paper for details!)
    rec_list = CoveoCartRecList(
        model=model,
        dataset=coveo_dataset
    )

    # invoke rec_list to run tests
    rec_list(verbose=True)

    # get the MovieLens 25M dataset as a RecDataset object
    movielens_dataset = MovieLensDataset()

    # re-use a skip-gram model from reclist to train a latent product space, to be used
    # (through knn) to build a recommender
    model = MovieLensP2VRecModel()
    model.train(movielens_dataset.x_train)

    # instantiate rec_list object, prepared with standard quantitative tests
    # and sensible behavioral tests (check the paper for details!)
    rec_list = MovieLensSimilarItemRecList(
        model=model,
        dataset=movielens_dataset
    )

    # invoke rec_list to run tests
    rec_list(verbose=True)



def test_mrr():
    """
    Testing MRR works as intended
    """
    list_a = [[0], [1]]
    list_b = [[0, 1], [1, 0]]
    list_c = [[2, 3], [0, 1]]

    assert mrr_at_k(list_b, list_a, 0) == 1
    assert mrr_at_k(list_c, list_a, 20) == 0.25
    assert mrr_at_k(list_c, list_a, 0) == 0

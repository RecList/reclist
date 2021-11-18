#!/usr/bin/env python

"""Tests for `reclist` package."""

import pytest


from reclist import reclist

from reclist.datasets import *
from reclist.rectest.standard_metrics import mrr_at_k

def test_basic_dataset_downloading():
    CoveoDataset()
    MovieLensDataset()
    SpotifyDataset()


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

#!/usr/bin/env python

"""Tests for `reclist` package."""

import random

import pandas as pd
from reclist.metrics.standard_metrics import hit_rate_at_k, mrr_at_k




def test_hits():

    df_b = pd.DataFrame(
        [[9, 0, 32, 12], [1, 0, 7, 12], [0, 1, 5, 12], [12, 32, 66, 99]]
    )
    df_a = pd.DataFrame([[0], [1], [5], [99]])
    df_c = pd.DataFrame(
        [
            [10000, 10000, 10000],
            [10000, 10000, 10000],
            [10000, 10000, 10000],
            [10000, 10000, 10000],
        ]
    )

    assert hit_rate_at_k(df_b, df_a, 5) == 1
    assert hit_rate_at_k(df_b, df_a, 100) == 1  # out of bounds
    assert hit_rate_at_k(df_b, df_a, 1) == 0.25  # 1 out of 4
    assert hit_rate_at_k(df_b, df_a, 2) == 0.5  # 2 out of 4
    assert hit_rate_at_k(df_b, df_a, 3) == 0.75  # 3 out of 4

    assert hit_rate_at_k(df_c, df_a, 3) == 0
    assert hit_rate_at_k(df_c, df_a, 100) == 0
    assert hit_rate_at_k(df_c, df_a, 5) == 0


def test_mrr():
    """
    Testing MRR works as intended
    """
    df_a = pd.DataFrame([[0], [1]])
    df_b = pd.DataFrame([[0, 1], [1, 0]])
    df_c = pd.DataFrame([[2, 3], [0, 1]])

    assert mrr_at_k(df_b, df_a, 1) == 1
    assert mrr_at_k(df_c, df_a, 2) == 0.25
    assert mrr_at_k(df_c, df_a, 1) == 0

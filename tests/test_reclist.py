#!/usr/bin/env python

"""Tests for `reclist` package."""

import random
import pytest
import pandas as pd
from reclist.metrics.standard_metrics import hit_rate_at_k, mrr_at_k




def test_hits():
    """
    Testing Hit Rate
    """

    df_b = pd.DataFrame(
        [[9, 0, 32, 12, ], 
         [1, 0, 7, 12], 
         [0, 1, 5, 12], 
         [12, 32, 66, 99]]
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
    df_d = pd.DataFrame(
        [[9, 0, 32, 12, None ], 
         [1, 0, 7, 12,  None], 
         [0, 1, 5, 12,  None], 
         [12, 32, 66, None,  None]]
    )
    df_e = pd.DataFrame(
        [[0,  32, None], 
         [10, 13, 4   ], 
         [5, None, None],
         [99, 0, 15, 66]]
    )

    assert hit_rate_at_k(df_b, df_a, 5) == 1
    assert hit_rate_at_k(df_b, df_a, 100) == 1  # out of bounds
    assert hit_rate_at_k(df_b, df_a, 1) == 0.25  # 1 out of 4
    assert hit_rate_at_k(df_b, df_a, 2) == 0.5  # 2 out of 4
    assert hit_rate_at_k(df_b, df_a, 3) == 0.75  # 3 out of 4
    assert hit_rate_at_k(df_d, df_a, 5) == 0.75  # handling None/NaN case

    # Multi target tests
    assert hit_rate_at_k(df_d, df_e, 5) == 0.75
    assert hit_rate_at_k(df_d, df_e, 100) == 0.75  # out of bounds
    assert hit_rate_at_k(df_d, df_e, 1) == 0.0  # 1 out of 4
    assert hit_rate_at_k(df_d, df_e, 2) == 0.25  # 2 out of 4
    assert hit_rate_at_k(df_d, df_e, 3) == 0.75  # 3 out of 4
    assert hit_rate_at_k(df_d, df_e, 5) == 0.75  # handling None/NaN case

    # not hit cases
    assert hit_rate_at_k(df_c, df_a, 3) == 0
    assert hit_rate_at_k(df_c, df_a, 100) == 0
    assert hit_rate_at_k(df_c, df_a, 5) == 0


def test_mrr():
    """
    Testing MRR
    """
    df_a = pd.DataFrame([[0], [1]])
    df_b = pd.DataFrame([[0, 1], [1, 0]])
    df_c = pd.DataFrame([[2, 3], [0, 1]])


    df_d = pd.DataFrame(
        [[0, 2, 14, 32, None],
         [1, 8,  7, None, None]]
    )
    df_e = pd.DataFrame(
        [[10, 12, 14, None, None, None], 
         [22, 8, 64, 13, 1, 0]]
    )
    df_f = pd.DataFrame(
        [[10, 12, 14, None, None, None], 
         [22, 1, 64, 13, 1, 0]]
    )

    df_g = pd.DataFrame(
        [[10, 12, 14, None, None, None], 
         [22, 17, 64, 13, 1, 0]]
    )
    # df_f = pd.DataFrame(
    #     [[2, 3], 
    #      [0, 1]]
    #     )
    
    # basic tests
    assert mrr_at_k(df_b, df_a, 1) == 1
    assert mrr_at_k(df_c, df_a, 2) == 0.25
    assert mrr_at_k(df_c, df_a, 1) == 0

    # multi target tests
    assert mrr_at_k(df_e, df_d, 2) == 1/4
    assert mrr_at_k(df_e, df_d, 3) == pytest.approx(5/12)
    assert mrr_at_k(df_e, df_d, 6) == pytest.approx(5/12)

    # k larger than pred size
    assert mrr_at_k(df_e, df_d, 20) == pytest.approx(5/12)
    
    # repeated prediction that is a hit
    assert mrr_at_k(df_f, df_d, 6) == pytest.approx(5/12)

    assert mrr_at_k(df_g, df_d, 6) == pytest.approx(4/15)
    
    
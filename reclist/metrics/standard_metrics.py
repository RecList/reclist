import collections
import itertools
import random

import numpy as np
import pandas as pd
from typing import Union, List, Dict


def hits_at_k(
    y_pred: pd.DataFrame, y_test:pd.DataFrame, k: int
) -> np.array:
    """
    N = number test cases
    M = number ground truth per test case
    returns array(N, preds_per_user, k)

    """
    y_test_mask = y_test.values != -1  # N x M

    y_pred_mask = y_pred.values[:, :k] != -1  # N x k

    y_test = y_test.values[:, :, None]  # N x M x 1
    y_pred = y_pred.values[:, None, :k]  # N x 1 x k

    hits = y_test == y_pred  # N x M x k
    hits = hits * y_test_mask[:, :, None]  # N x M x k
    hits = hits * y_pred_mask[:, None, :]  # N x M x k

    return hits


def ranks_at_k(y_pred: pd.DataFrame, y_test:pd.DataFrame, k: int) -> np.array:
    """
    N = number test cases
    M = number ground truth per test case
    returns shape (N, preds_per_user)
    """
    hits = hits_at_k(y_pred, y_test, k)  # N x M x k
    ranks = hits * np.arange(1, k + 1, 1)[None, None, :]  # N x M x k
    ranks = ranks.max(axis=2)  # N x M
    return ranks


def misses_at_k(
    y_pred: pd.DataFrame, y_test: pd.DataFrame, k: int
) -> np.array:
    """_summary_

    Args:
        y_pred (pd.DataFrame): predictions
        y_test (pd.DataFrame): ground truth
        k (int): top k
    Returns:
        np.array: same shape as hits_at_k
    """
    hits = hits_at_k(y_pred, y_test, k)  # N x M x k
    return 1 - hits


def hit_rate_at_k(y_pred: pd.DataFrame, y_test: pd.DataFrame, k: int) -> float:
    """
    N = number test cases
    M = number ground truth per test case
    """
    hits = hits_at_k(y_pred, y_test, k)  # N x M x k
    hits = hits.max(axis=1)  # N x k
    return hits.max(axis=1).mean()  # 1


def rr_at_k(y_pred: pd.DataFrame, y_test: pd.DataFrame, k: int)-> np.array:
    """
    returns shape (N,)
    """
    ranks = ranks_at_k(y_pred, y_test, k).astype(np.float64)  # N x M
    reciprocal_ranks = np.reciprocal(ranks, out=ranks, where=ranks > 0)  # N x M
    return reciprocal_ranks.max(axis=1)  # N


def mrr_at_k(y_pred: pd.DataFrame, y_test: pd.DataFrame, k: int) -> float:
    return rr_at_k(y_pred, y_test, k=k).mean()


####################### LIST-WISE CODE #####################################


def statistics(x_train, y_train, x_test, y_test, y_pred):
    train_size:int = len(x_train)
    test_size:int = len(x_test)
    # num non-zero preds
    num_preds:int = len([p for p in y_pred if p])
    return {
        "training_set__size": train_size,
        "test_set_size": test_size,
        "num_non_null_predictions": num_preds,
    }


def sample_hits_at_k(y_preds:List[List[str]], y_test:List[List[str]], x_test=None, k:int=3, size:int=3)-> List:
    hits = []
    for idx, (_p, _y) in enumerate(zip(y_preds, y_test)):
        if _y[0] in _p[:k]:
            hit_info = {
                "Y_TEST": [_y[0]],
                "Y_PRED": _p[:k],
            }
            if x_test:
                hit_info["X_TEST"] = [x_test[idx][0]]
            hits.append(hit_info)

    if len(hits) < size or size == -1:
        return hits
    return random.sample(hits, k=size)


def sample_misses_at_k(y_preds:List[List[str]], y_test:List[List[str]], x_test=None, k:int=3, size:int=3)-> List[Dict]:
    '''
    returns a list of dicts with the following keys:
    Y_TEST: the ground truth
    Y_PRED: the predictions
    '''
    misses = []
    for idx, (_p, _y) in enumerate(zip(y_preds, y_test)):
        if _y[0] not in _p[:k]:
            miss_info = {
                "Y_TEST": [_y[0]],
                "Y_PRED": _p[:k],
            }
            if x_test:
                miss_info["X_TEST"] = [x_test[idx][0]]
            misses.append(miss_info)

    if len(misses) < size or size == -1:
        return misses
    return random.sample(misses, k=size)


def hit_rate_at_k_nep(y_preds:List[str], y_test:List[str], k:int=3)-> float:
    """
    1d array of predictions
    1d array of ground truth
    """
    
    y_test = [[k] for k in y_test]
    return hit_rate_at_k_list(y_preds, y_test, k=k)


def hit_rate_at_k_list(y_preds:List[List[str]], y_test:List[List[str]], k:int=3) -> float:
    hits = 0
    for _p, _y in zip(y_preds, y_test):
        if len(set(_p[:k]).intersection(set(_y))) > 0:
            hits += 1
    return hits / len(y_test)


def mrr_at_k_nep_list(y_preds:List[str], y_test:List[str], k:int=3)-> float:
    """
    Computes MRR

    :param y_preds: predictions, as lists of lists
    :param y_test: target data, as lists of lists (eventually [[sku1], [sku2],...]
    :param k: top-k
    """
    y_test = [[k] for k in y_test]
    return mrr_at_k_list(y_preds, y_test, k=k)


def mrr_at_k_list(y_preds:List[List[str]], y_test:List[List[str]], k:int=3)-> float:
    """
    Computes MRR

    :param y_preds: predictions, as lists of lists
    :param y_test: target data, as lists of lists (eventually [[sku1], [sku2],...]
    :param k: top-k
    """
    rr = []
    for _p, _y in zip(y_preds, y_test):
        for rank, p in enumerate(_p[:k], start=1):
            if p in _y:
                rr.append(1 / rank)
                break
        else:
            rr.append(0)
    assert len(rr) == len(y_preds)
    return np.mean(rr)


def coverage_at_k(y_preds:List[List[str]], product_data:Dict, k:int=3)-> float:
    """
    product_data: dict with product ids as keys
    """
    pred_skus = set(itertools.chain.from_iterable(y_preds[:k]))
    all_skus = set(product_data.keys())
    nb_overlap_skus = len(pred_skus.intersection(all_skus))

    return nb_overlap_skus / len(all_skus)


def popularity_bias_at_k(y_preds:List[List[str]], x_train:List[List[str]], k:int=3)-> float:
    # estimate popularity from training data
    pop_map:Dict = collections.defaultdict(lambda: 0)
    num_interactions = 0
    for session in x_train:
        for event in session:
            pop_map[event] += 1
            num_interactions += 1
    # normalize popularity
    pop_map = {k: v / num_interactions for k, v in pop_map.items()}
    all_popularity = []
    for p in y_preds:
        average_pop = (
            sum(pop_map.get(_, 0.0) for _ in p[:k]) / len(p) if len(p) > 0 else 0
        )
        all_popularity.append(average_pop)
    return sum(all_popularity) / len(y_preds)


def precision_at_k(y_preds:List[List[str]], y_test:List[List[str]], k:int=3)-> float:
    precision_ls = [
        len(set(_y).intersection(set(_p[:k]))) / len(_p[:k]) if _p else 1
        for _p, _y in zip(y_preds, y_test)
    ]
    return np.average(precision_ls)


def recall_at_k(y_preds:List[List[str]], y_test:List[List[str]], k:int=3)-> float:
    recall_ls = [
        len(set(_y).intersection(set(_p[:k]))) / len(_y) if _y else 1
        for _p, _y in zip(y_preds, y_test)
    ]
    return np.average(recall_ls)

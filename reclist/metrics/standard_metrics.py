"""
Implement standard recommender system retrieval metrics leveraging numpy/pandas backend.
"""
import collections
import itertools
import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from typing import Union


def func_per_slice(y_test, y_pred, categories, func):
    labels = set(categories)

    c = []
    global_score = func(y_test, y_pred)

    for label in labels:
        mask = np.array([True if k == label else False for k in categories])
        local_score = func(y_test[mask], y_pred[mask])
        c.append(np.abs(local_score - global_score))
    return np.mean(c)

def accuracy_per_slice(y_test, y_pred, categories):

    y_test = np.array(y_test)
    y_pred = np.array(y_pred)
    return func_per_slice(y_test, y_pred, categories, accuracy_score)

def hits_at_k(
    y_pred: pd.DataFrame, 
    y_test: pd.DataFrame,
    k: int
) -> Union[np.array, pd.DataFrame]:
    """
    Computes whether for the n-th test case, the m-th ground truth is a hit with the k-th prediction.

    Parameters
    ----------
    y_pred: pd.DataFrame
        Array of predictions with shape N x k (?)
    y_test: pd.DataFrame
        Array of ground truth with shape N x m
    k: int
        Index cut-off for prediciton 
    
    Returns
    -------
    out : np.array, pd.DataFrame
         array indicicating whether for the n-th test case, the m-th ground truth is a hit with the k-th prediction

    Examples
    --------
    >>> y_pred = pd.DataFrame([[1, 2, 4], [3, 6, 2]])
    >>> y_test = pd.DataFrame([[5, 1, 0], [6, 2, 3]])
    >>> hits_at_k(y_pred, y_test, k=2)
    array([
        [[False, False],
         [True, False],
         [False, False]],
        [[False, True],
         [False, False],
         [True, False]]
    ])
    Explanation:
    
    Considering the first test case [5, 1, 0].
    Note: The y_pred becomes pd.DataFrame([[1, 2], [3, 6]]) because k=2. 

    We now focus on the first predicted set, [1, 2].

    How to interpret the output:
        array[0][0][0] is False because our first element predicted (1) is not equal to the first element of the test case (5).
        array[0][0][1] is False because our second element predicted (2) is not equal to the first element of the test case (5).
        array[0][1][0] is True because our first element predicted (2) is equal to the second element of the test case (1).
    """
    
    y_test_mask = ~y_test.isna().values         # N x M

    y_pred_mask = ~y_pred.isna().values[:, :k]  # N x k

    y_test = y_test.values[:, :, None]        # N x M x 1
    y_pred = y_pred.values[:, None, :k]       # N x 1 x k

    hits = y_test == y_pred                   # N x M x k
    hits = hits * y_test_mask[:, :, None]     # N x M x k
    hits = hits * y_pred_mask[:, None, :]     # N x M x k

    return hits


def ranks_at_k(
    y_pred: pd.DataFrame, 
    y_test: pd.DataFrame,
    k: int
) -> Union[np.array, pd.DataFrame]:
    """
    Computes for every test case, n, the rank of the m-th ground truth label out of top-k predictions.
    Rank is 0 if ground truth is not in prediction.

    Parameters
    ----------
    y_pred: pd.DataFrame
        Array of predictions with shape N x k (?)
    y_test: pd.DataFrame
        Array of ground truth with shape N x m
    k: int
        Index cut-off for prediciton 
    
    Returns
    -------
    out : np.array, pd.DataFrame
         array indicicating for each test case n, the rank of the m-th ground truth label in top-k predictions.

    Examples
    --------
    >>> y_pred = pd.DataFrame([[1, 2, 4], [3, 6, 2]])
    >>> y_test = pd.DataFrame([[5, 1, 0], [6, 2, 3]])
    >>> hits_at_k(y_pred, y_test, k=2)
    array([[0, 1, 0],
           [2, 0, 1]])

    """
    hits = hits_at_k(y_pred, y_test, k)                     # N x M x k
    # TODO: hits_at_k can be modified to return df with last dim=k instead of preds shape
    rank_overlap = min(k, hits.shape[-1])
    ranks = hits * np.arange(1, rank_overlap + 1, 1)[None, None, :]    # N x M x k
    # set to float
    ranks = ranks.astype(float)
    # set non-hits to infinity
    ranks[ranks==0] = np.inf
    # get highest rank; if no hit, rank is infinite
    ranks = ranks.min(axis=2)                               # N x M
    return ranks


def misses_at_k(
    y_pred: pd.DataFrame,
    y_test: pd.DataFrame,
    k: int
) -> Union[np.array, pd.DataFrame]:
    """
    Computes whether for the n-th test case, the m-th ground truth is a miss with the k-th prediction.

    Parameters
    ----------
    y_pred: pd.DataFrame
        Array of predictions with shape N x k (?)
    y_test: pd.DataFrame
        Array of ground truth with shape N x m
    k: int
        Index cut-off for prediciton 
    
    Returns
    -------
    out : np.array, pd.DataFrame
         array indicicating whether for the n-th test case, the m-th ground truth is a miss with the k-th prediction

    Examples
    --------
    >>> y_pred = pd.DataFrame([[1, 2, 4], [3, 6, 2]])
    >>> y_test = pd.DataFrame([[5, 1, 0], [6, 2, 3]])
    >>> misses_at_k(y_pred, y_test, k=2)
    array([
        [[True, True],
         [False, True],
         [True, True]],
        [[True, False],
         [True, True],
         [False, True]]
    ])
    """  

    hits = hits_at_k(y_pred, y_test, k)  # N x M x k
    return ~hits


def hit_rate_at_k(
    y_pred: pd.DataFrame,
    y_test: pd.DataFrame,
    k: int
) -> float:
    """
    Computes the hit rate @ k.

    Parameters
    ----------
    y_pred: pd.DataFrame
        Array of predictions with shape N x k (?)
    y_test: pd.DataFrame
        Array of ground truth with shape N x m
    k: int
        Index cut-off for prediciton 
    
    Returns
    -------
    out : float
         hit rate @ k

    Examples
    --------
    >>> y_pred = pd.DataFrame([[1, 2, 4], [3, 6, 2]])
    >>> y_test = pd.DataFrame([[5, 1, 0], [1, 2, 0]])
    >>> hit_rate_at_k(y_pred, y_test, k=2)
    0.5
    """
    assert len(y_pred.shape) == 2
    assert len(y_test.shape) == 2
    assert y_pred.shape[0] == y_test.shape[0]

    hits = hits_at_k(y_pred, y_test, k)  # N x M x k
    hits = hits.max(axis=1)              # N x k
    return hits.max(axis=1).mean()       # 1


def rr_at_k(
    y_pred: pd.DataFrame,
    y_test: pd.DataFrame,
    k: int
) -> Union[np.array, pd.DataFrame]:
    """
    Computes the reciprocal rank of ground truth in the top-k predictions for the n-th test case.

    Parameters
    ----------
    y_pred: pd.DataFrame
        Array of predictions with shape N x k (?)
    y_test: pd.DataFrame
        Array of ground truth with shape N x m
    k: int
        Index cut-off for prediciton 
    
    Returns
    -------
    out : np.array, pd.DataFrame
        array indicicating, for the n-th test case, the reciprocal rank of the m-th ground truth in the k predictions.

    Examples
    --------
    >>> y_pred = pd.DataFrame([[2, 1, 4], [3, 6, 2]])
    >>> y_test = pd.DataFrame([[5, 1, 0], [2, 1, 0]])
    >>> rr_at_k(y_pred, y_test, k=2)
    array([0.5, 0.0])
    """
    
    ranks = ranks_at_k(y_pred, y_test, k).astype(np.float64)  # N x M
    reciprocal_ranks = np.reciprocal(ranks, out=ranks,)       # N x M
    # reciprocal_ranks = np.reciprocal(ranks, out=ranks, where=ranks > 0)  # N x M
    return reciprocal_ranks.max(axis=1)  # N


def mrr_at_k(
    y_pred: pd.DataFrame, 
    y_test: pd.DataFrame, 
    k: int
) -> float:
    
    """
    Computes the mean reciprocal rank @ k (mrr @ k)

    Parameters
    ----------
    y_pred: pd.DataFrame
        Array of predictions with shape N x k (?)
    y_test: pd.DataFrame
        Array of ground truth with shape N x m
    k: int
        Index cut-off for prediciton 
    
    Returns
    -------
    out : float
         mean reciprocal rank @ k

    Examples
    --------
    >>> y_pred = pd.DataFrame([[2, 1, 4], [3, 6, 2]])
    >>> y_test = pd.DataFrame([[5, 1, 0], [2, 1, 0]])
    >>> mrr_at_k(y_pred, y_test, k=2)
    0.25
    """
    assert len(y_pred.shape) == 2
    assert len(y_test.shape) == 2
    assert y_pred.shape[0] == y_test.shape[0]

    return rr_at_k(y_pred, y_test, k=k).mean()


def precision_at_k(
    y_pred: pd.DataFrame, 
    y_test: pd.DataFrame, 
    k: int
):
    """
    Computes the precision @ k

    Parameters
    ----------
    y_pred: pd.DataFrame
        Array of predictions with shape N x k (?)
    y_test: pd.DataFrame
        Array of ground truth with shape N x m
    k: int
        Index cut-off for prediciton 
    
    Returns
    -------
    out : float
         precision @ k

    Examples
    --------
    >>> y_pred = pd.DataFrame([[2, 1, 4], [0, 1, 2]])
    >>> y_test = pd.DataFrame([[5, 1, 0], [2, 1, 0]])
    >>> precision_at_k(y_pred, y_test, k=2)
    0.75
    """
    hits = hits_at_k(y_pred, y_test, k=k)   # N x M x k
    hits = hits.max(axis=1)
    num_intersection = hits.sum(axis=1)
    num_preds = (~y_pred.isna().values)[:, :k].sum(axis=1)

    return np.average(np.divide(num_intersection, num_preds, where=num_preds>0))
    

def recall_at_k(
    y_pred: pd.DataFrame, 
    y_test: pd.DataFrame, 
    k: int
):
    """
    Computes the recall @ k

    Parameters
    ----------
    y_pred: pd.DataFrame
        Array of predictions with shape N x k (?)
    y_test: pd.DataFrame
        Array of ground truth with shape N x m
    k: int
        Index cut-off for prediciton 
    
    Returns
    -------
    out : float
         recall @ k

    Examples
    --------
    >>> y_pred = pd.DataFrame([[2, 1, 4], [3, 6, 2]])
    >>> y_test = pd.DataFrame([[5, 1, 0], [2, 1, 0]])
    >>> recall_at_k(y_pred, y_test, k=2)
    0.3333
    """
    
    hits = hits_at_k(y_pred, y_test, k=k)   # N x M x k
    hits = hits.max(axis=1)
    num_intersection = hits.sum(axis=1)
    num_targets = (~y_test.isna().values).sum(axis=1)

    return np.average(np.divide(num_intersection, num_targets, where=num_targets>0))
    

def statistics(x_train, y_train, x_test, y_test, y_pred):
    train_size = x_train.shape[0]
    test_size = x_test.shape[0]
    # num non-zero preds
    num_preds = (~y_pred.isna().values).max(axis=1).sum()
    return {
        "training_set_size": train_size,
        "test_set_size": test_size,
        "num_non_null_predictions": num_preds,
    }


####################### LIST-WISE CODE #####################################

def sample_hits_at_k(y_preds, y_test, x_test=None, k=3, size=3):
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


def sample_misses_at_k(y_preds, y_test, x_test=None, k=3, size=3):
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


def coverage_at_k(y_preds, product_data, k=3):
    pred_skus = set(itertools.chain.from_iterable(y_preds[:k]))
    all_skus = set(product_data.keys())
    nb_overlap_skus = len(pred_skus.intersection(all_skus))

    return nb_overlap_skus / len(all_skus)


def popularity_bias_at_k(y_preds, x_train, k=3):
    # estimate popularity from training data
    pop_map = collections.defaultdict(lambda: 0)
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

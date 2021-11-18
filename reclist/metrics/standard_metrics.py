import random
import itertools
import numpy as np


def statistics(x_train, y_train, x_test, y_test, y_pred):
    train_size = len(x_train)
    test_size = len(x_test)
    # num non-zero preds
    num_preds = len([p for p in y_pred if p])
    return {
        'train_size': train_size,
        'test_size': test_size,
        'num_preds': num_preds
    }


def sample_hits_at_k(y_preds, y_test, k=3, size=3):
    hits = []
    for _p, _y in zip(y_preds, y_test):
        if _y[0] in _p[:k]:
            hits.append({'Y_TEST': [_y[0]], 'Y_PRED': _p[:k]})
    if len(hits) < size or size == -1:
        return hits
    return random.sample(hits, k=size)


def sample_misses_at_k(y_preds, y_test, k=3, size=3):
    misses = []
    for _p, _y in zip(y_preds, y_test):
        if _y[0] not in _p[:k]:
            misses.append({'Y_TEST': [_y[0]], 'Y_PRED': _p[:k]})
    if len(misses) < size or size == -1:
        return misses
    return random.sample(misses, k=size)


def hit_rate_at_k(y_preds, y_test, k=3):
    hits = 0
    for _p, _y in zip(y_preds, y_test):
        if _y[0] in _p[:k]:
            hits += 1
    return hits / len(y_test)


def mrr_at_k(y_preds, y_test, k=3):
    """
    Computes MRR

    :param y_preds: predictions, as lists of lists
    :param y_test: target data, as lists of lists (eventually [[sku1], [sku2],...]
    :param k: top-k
    """
    rr = []
    y_test = [k[0] for k in y_test]
    for _p, _y in zip(y_preds, y_test):
        if _y in _p[:k+1]:
            rank = _p[:k+1].index(_y) + 1
            rr.append(1/rank)
        else:
            rr.append(0)
    return np.mean(rr)


def coverage_at_k(y_preds, product_data, k=3):
    pred_skus = set(itertools.chain.from_iterable(y_preds[:k]))
    all_skus = set(product_data.keys())
    nb_overlap_skus = len(pred_skus.intersection(all_skus))

    return nb_overlap_skus / len(all_skus)

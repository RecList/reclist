from collections import Counter, defaultdict
import numpy as np
import math
from reclist.metrics.standard_metrics import hit_rate_at_k, sample_hits_at_k, sample_misses_at_k
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from reclist.current import current


def hits_distribution_by_rating(y_test, y_preds, debug=False):
    """
    Calculates the distribution of hit-rate across movie ratings in testing data
    """
    hits = defaultdict(int)
    total = defaultdict(int)

    for target, pred in zip(y_test, y_preds):
        target_movie_id, target_rating = target[0]["movieId"], target[0]["rating"]
        for movie in pred:
            pred_movie_id, _ = movie["movieId"], movie["rating"]
            if target_movie_id == pred_movie_id:
                hits[target_rating] += 1
                break
        total[target_rating] += 1
    hits_distribution_by_rating = {}
    for target_rating in sorted(hits.keys()):
        hit_rate = hits[target_rating] / total[target_rating]
        hits_distribution_by_rating[target_rating] = hit_rate

    if debug:
        x_tick_names = list(hits_distribution_by_rating.keys())
        x_tick_idx = list(range(len(x_tick_names)))
        plt.figure(dpi=150)
        plt.bar(
            x_tick_idx,
            [hit_rate for hit_rate in hits_distribution_by_rating.values()],
            align='center'
        )
        plt.xticks(
            list(range(len(hits_distribution_by_rating))), x_tick_names, fontsize=10
        )
        plt.savefig(os.path.join(current.report_path,
                                'plots',
                                'hit_distribution_rating.png'))
        plt.clf()

    return hits_distribution_by_rating

def roundup(x: int):
    div = 10.0 ** (len(str(x)))
    return int(math.ceil(x / div)) * div


def hits_distribution(x_train, x_test, y_test, y_preds, k=3, debug=False):
    # get product interaction frequency
    prod_interaction_cnt = Counter([_ for x in x_train for _ in x])
    hit_per_interaction_cnt = defaultdict(list)
    for _x, _y_test, _y_pred in zip(x_test, y_test, y_preds):
        _x_cnt = prod_interaction_cnt[_x[0]]
        # TODO: allow for generic metric
        hit_per_interaction_cnt[_x_cnt].append(hit_rate_at_k([_y_pred], [_y_test], k=k))
    # get max product frequency
    max_cnt = prod_interaction_cnt.most_common(1)[0][1]
    # round up to nearest place
    max_cnt = int(roundup(max_cnt))
    # generate log-bins
    indices = np.logspace(1, np.log10(max_cnt), num=int(np.log10(max_cnt))).astype(np.int64)
    indices = np.concatenate(([0], indices))
    counts_per_bin = [[_ for i in range(low, high) for _ in hit_per_interaction_cnt[i]]
                      for low, high in zip(indices[:-1], indices[1:])]

    histogram = [np.mean(counts) if counts else 0 for counts in counts_per_bin]
    count = [len(counts) for counts in counts_per_bin]

    if debug:
        # debug / visualization
        plt.bar(indices[1:], histogram, width=-np.diff(indices) / 1.05, align='edge')
        plt.xscale('log', base=10)
        plt.title('HIT Distribution Across Product Frequency')
        # plt.show()
        plt.savefig(os.path.join(current.report_path, 'plots', 'hit_distribution.png'))
        plt.clf()

    return {
        'histogram': {int(k): v for k, v in zip(indices[1:], histogram)},
        'counts': {int(k): v for k, v in zip(indices[1:], count)}
    }


def hits_distribution_by_slice(slices,
                               y_test,
                               y_preds,
                               k=3,
                               sample_size=3,
                               debug=False):
    hit_rate_per_slice = defaultdict(dict)
    for slice_name, slice_idx in slices.items():
        # get predictions for slice
        slice_y_preds = [y_preds[_] for _ in slice_idx]
        # get labels for slice
        slice_y_test = [y_test[_] for _ in slice_idx]
        # TODO: We may want to allow for generic metric to be used here
        slice_hr = hit_rate_at_k(slice_y_preds, slice_y_test, k=k)
        # store results
        hit_rate_per_slice[slice_name]['hit_rate'] = slice_hr
        # hit_rate_per_slice[slice_name]['hits'] = sample_hits_at_k(slice_y_preds, slice_y_test, k=k, size=sample_size)
        # hit_rate_per_slice[slice_name]['misses'] = sample_misses_at_k(slice_y_preds, slice_y_test, k=k, size=sample_size)

    # debug / visualization
    if debug:
        x_tick_names = list(hit_rate_per_slice.keys())
        x_tick_idx = list(range(len(x_tick_names)))
        plt.figure(dpi=150)
        plt.bar(x_tick_idx, [_['hit_rate'] for _ in hit_rate_per_slice.values()], align='center')
        plt.xticks(list(range(len(hit_rate_per_slice))), x_tick_names, rotation=45, fontsize=5)
        plt.savefig(os.path.join(current.report_path,
                                 'plots',
                                 'hit_distribution_slice.png'))
        plt.clf()

    # cast to normal dict
    return dict(hit_rate_per_slice)

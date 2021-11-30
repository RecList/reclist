from collections import Counter, defaultdict
from reclist.metrics.standard_metrics import hit_rate_at_k
import matplotlib.pyplot as plt
import numpy as np
import math
import os
from reclist.current import current


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
        plt.bar(indices[1:], histogram, width=-np.diff(indices)/1.05, align='edge')
        plt.xscale('log', base=10)
        plt.title('HIT Distribution Across Product Frequency')
        # plt.show()
        plt.savefig(os.path.join(current.report_path,'plots','hit_distribution.png'))
        plt.clf()

    return {
             'histogram': {int(k): v for k, v in zip(indices[1:], histogram)},
             'counts':  {int(k): v for k, v in zip(indices[1:], count)}
           }

from collections import Counter, defaultdict
from reclist.metrics.standard_metrics import hit_rate_at_k
import matplotlib.pyplot as plt
import numpy as np
import math


def roundup(x: int):
    div = 10.0 ** (len(str(x)))
    return int(math.ceil(x / div)) * div


def hits_distribution(x_train, x_test, y_test, y_preds, k=3, debug=False):

    prod_interaction_cnt = Counter([_ for x in x_train for _ in x])
    hit_per_interaction_cnt = defaultdict(list)
    for _x, _y_test, _y_pred in zip(x_test, y_test, y_preds):
        _x_cnt = prod_interaction_cnt[_x[0].split('_')[0]]
        # TODO: We may want to allow for generic metric to be used here
        hit_per_interaction_cnt[_x_cnt].append(hit_rate_at_k([_y_pred], [_y_test], k=k))

    max_cnt = prod_interaction_cnt.most_common(1)[0][1]
    # round up to nearest place
    max_cnt = int(roundup(max_cnt))
    indices = np.logspace(1, np.log10(max_cnt), num=int(np.log10(max_cnt))).astype(np.int64)
    # start from 0
    indices = [0] + list(indices)
    histogram = [np.mean([_ for i in range(low, high) for _ in hit_per_interaction_cnt[i]])
                 for low, high in zip(indices[:-1], indices[1:])]
    count = [len([_ for i in range(low, high) for _ in hit_per_interaction_cnt[i]])
                 for low, high in zip(indices[:-1], indices[1:])]

    if debug:
        # debug / visualization
        plt.bar(indices[1:],histogram, width=-np.diff(indices)/1.05, align='edge')
        plt.xscale('log', base=10)
        plt.title('HIT distribution across prod frequency')
        plt.show()

    return {
             'histogram': {int(k): v for k, v in zip(indices[1:], histogram)},
             'counts':  {int(k): v for k, v in zip(indices[1:], count)}
           }

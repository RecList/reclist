from collections import defaultdict
import os

import matplotlib.pyplot as plt

from reclist.current import current


def hits_distribution_by_rating(y_test, y_preds, debug=False):
    """
    Calculates the distribution of hit-rate across the movie ratings in testing data
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
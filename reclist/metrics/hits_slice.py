from collections import defaultdict
from reclist.metrics.standard_metrics import hit_rate_at_k, sample_hits_at_k, sample_misses_at_k
import matplotlib.pyplot as plt
import os
from reclist.current import current


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
        plt.bar(x_tick_idx,[_['hit_rate'] for _ in hit_rate_per_slice.values()], align='center')
        plt.xticks(list(range(len(hit_rate_per_slice))), x_tick_names, rotation=45, fontsize=5)
        plt.savefig(os.path.join(current.report_path,
                                 'plots',
                                 'hit_distribution_slice.png'))
        plt.clf()

    # cast to normal dict
    return dict(hit_rate_per_slice)

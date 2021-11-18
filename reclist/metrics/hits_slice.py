from collections import Counter, defaultdict
from reclist.metrics.standard_metrics import hit_rate_at_k, sample_hits_at_k, sample_misses_at_k
import matplotlib.pyplot as plt


def hits_distribution_by_slice(slice_fns: dict,
                               y_test,
                               y_preds,
                               product_data,
                               k=3,
                               sample_size=3,
                               debug=False):

    hit_rate_per_slice = defaultdict(dict)
    for slice_name, filter_fn in slice_fns.items():
        # get indices for slice
        slice_idx = [idx for idx,_y in enumerate(y_test) if _y[0] in product_data and filter_fn(product_data[_y[0]])]
        # get predictions for slice
        slice_y_preds = [y_preds[_] for _ in slice_idx]
        # get labels for slice
        slice_y_test = [y_test[_] for _ in slice_idx]
        # TODO: We may want to allow for generic metric to be used here
        slice_hr = hit_rate_at_k(slice_y_preds, slice_y_test,k=3)
        # store results
        hit_rate_per_slice[slice_name]['hit_rate'] = slice_hr
        hit_rate_per_slice[slice_name]['hits'] = sample_hits_at_k(slice_y_preds, slice_y_test, k=k, size=sample_size)
        hit_rate_per_slice[slice_name]['misses'] = sample_misses_at_k(slice_y_preds, slice_y_test, k=k, size=sample_size)

    # debug / visualization
    if debug:
        x_tick_names = list(hit_rate_per_slice.keys())
        x_tick_idx = list(range(len(x_tick_names)))
        plt.bar(x_tick_idx, hit_rate_per_slice.values(), align='center')
        plt.xticks(list(range(len(hit_rate_per_slice))), x_tick_names)
        plt.show()

    # cast to normal dict
    return dict(hit_rate_per_slice)

import random
import numpy as np
from tqdm import tqdm


def session_perturbation_test(model,
                              x_test,
                              y_preds,
                              perturbation_fn,
                              id_fn,
                              k):
    overlap_ratios = []
    y_p = []
    x_perturbs = []
    # generate a batch of perturbations
    for _x, _y_p in zip(x_test, y_preds):
        # perturb last item in session
        x_perturb = perturbation_fn(_x['tracks'])
        if not x_perturb:
            continue
        x_perturbs.append({'tracks': x_perturb})
        y_p.append(_y_p)
    # make predictions over perturbed inputs
    y_perturbs = model.predict(x_perturbs)
    # extract uri
    y_p, y_perturbs = id_fn(y_p), id_fn(y_perturbs)
    # check for overlapping predictions
    for _y_p, _y_perturb in zip(y_p, y_perturbs):
        if _y_p and _y_perturb:
            # compute prediction intersection
            intersection = set(_y_perturb[:k]).intersection(_y_p[:k])
            overlap_ratio = len(intersection) / len(_y_p[:k])
            overlap_ratios.append(overlap_ratio)
        else:
            overlap_ratios.append(0)

    return np.mean(overlap_ratios)

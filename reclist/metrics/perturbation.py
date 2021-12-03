import numpy as np


def session_perturbation_test(model,
                              x_test,
                              y_preds,
                              perturbation_fn,
                              id_fn,
                              k):
    # generate perturbations
    perturbed_pairs = [(perturbation_fn(_x), _y_p) for _x, _y_p in zip(x_test, y_preds) if perturbation_fn(_x)]
    # extract perturbed x and original y
    x_test_p, y_preds_o = zip(*perturbed_pairs)
    # make new predictions over perturbed inputs
    y_preds_n = model.predict(x_test_p)
    # extract atomic unit for comparison
    y_preds_o, y_preds_n = id_fn(y_preds_o), id_fn(y_preds_n)
    # check for overlapping predictions
    overlap_ratios = []
    for _y_p_n, _y_p_o in zip(y_preds_o, y_preds_n):
        if _y_p_n and _y_p_o:
            # compute prediction intersection
            intersection = set(_y_p_o[:k]).intersection(_y_p_n[:k])
            overlap_ratio = len(intersection) / len(_y_p_n[:k])
            overlap_ratios.append(overlap_ratio)
        elif _y_p_n or _y_p_o:
            overlap_ratios.append(0)
    return np.mean(overlap_ratios)

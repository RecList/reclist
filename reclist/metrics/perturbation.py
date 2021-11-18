import random
import numpy as np
from tqdm import tqdm

# TODO: We might need to enforce some standardization of how CATEGORY is represented
def get_item_with_category(product_data: dict, category: set, to_ignore=None):
    to_ignore = [] if to_ignore is None else to_ignore
    skus = [_ for _ in product_data if product_data[_]['category_hash'] == category and _ not in to_ignore]
    if skus:
        return random.choice(skus)
    return []


def perturb_session(session, product_data):
    last_item = session[-1]
    if last_item not in product_data:
        return []
    last_item_category = product_data[last_item]['category_hash']
    similar_item = get_item_with_category(product_data, last_item_category, to_ignore=[last_item])
    if similar_item:
        new_session = session[:-1] + [similar_item]
        return new_session
    return []


def session_perturbation_test(model, x_test, y_preds, product_data):
    overlap_ratios = []
    # print(product_data)
    y_p = []
    s_perturbs = []

    # generate a batch of perturbations
    for idx, (s, _y_p) in enumerate(tqdm(zip(x_test,y_preds))):
        # perturb last item in session
        s = [ _.split('_')[0] for _ in s]
        s_perturb = perturb_session(s, product_data)
        if not s_perturb:
            continue

        s_perturb = ['_'.join([_,'add']) for _ in s_perturb]

        s_perturbs.append(s_perturb)
        y_p.append(_y_p)

    y_perturbs = model.predict(s_perturbs)

    for _y_p, _y_perturb in zip(y_p, y_perturbs):
        if _y_p and _y_perturb:
            # compute prediction intersection
            intersection = set(_y_perturb).intersection(_y_p)
            overlap_ratio = len(intersection)/len(_y_p)
            overlap_ratios.append(overlap_ratio)

    return np.mean(overlap_ratios)

import numpy as np
from reclist.metrics.standard_metrics import sample_misses_at_k
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from statistics import mean
from reclist.current import current
import os

def error_by_cosine_distance(model, y_test, y_preds, k=3, bins=25, debug=False):
    if not(hasattr(model.__class__, 'get_vector') and callable(getattr(model.__class__, 'get_vector'))):
        error_msg = "Error : Model {} does not support retrieval of vector embeddings".format(model.__class__)
        print(error_msg)
        return error_msg
    misses = sample_misses_at_k(y_preds, y_test, k=k, size=-1)
    cos_distances = []
    for m in misses:
        if m['Y_PRED']:
            vector_test = model.get_vector(m['Y_TEST'][0])
            vector_pred = model.get_vector(m['Y_PRED'][0])
            if vector_pred and vector_test:
                cos_dist = cosine(vector_pred, vector_test)
                cos_distances.append(cos_dist)

    histogram = np.histogram(cos_distances, bins=bins, density=False)
    # cast to list
    histogram = (histogram[0].tolist(), histogram[1].tolist())
    # debug / viz
    if debug:
        plt.hist(cos_distances, bins=bins)
        plt.title('dist over cosine distance prod space')
        plt.savefig(os.path.join(current.report_path,
                                 'plots',
                                 'distance_to_predictions.png'))
        plt.clf()
        # plt.show()

    return {'mean': np.mean(cos_distances), 'histogram': histogram}

def distance_to_query(model, x_test, y_test, y_preds, k=3, bins=25, debug=False):
    if not(hasattr(model.__class__, 'get_vector') and callable(getattr(model.__class__, 'get_vector'))):
        error_msg = "Error : Model {} does not support retrieval of vector embeddings".format(model.__class__)
        print(error_msg)
        return error_msg
    misses = sample_misses_at_k(y_preds, y_test, x_test=x_test, k=k, size=-1)
    x_to_y_cos = []
    x_to_p_cos = []
    for m in misses:
        if m['Y_PRED']:
            vector_x = model.get_vector(m['X_TEST'][0])
            vector_y = model.get_vector(m['Y_TEST'][0])
            vectors_p = [model.get_vector(_) for _ in m['Y_PRED']]
            c_dists =[]
            if not vector_x or not vector_y:
                continue
            x_to_y_cos.append(cosine(vector_x, vector_y))
            for v_p in vectors_p:
                if not v_p:
                    continue
                cos_dist = cosine(v_p, vector_x)
                if cos_dist:
                    c_dists.append(cos_dist)
            if c_dists:
                x_to_p_cos.append(mean(c_dists))

    h_xy = np.histogram(x_to_y_cos, bins=bins, density=False)
    h_xp = np.histogram(x_to_p_cos, bins=bins, density=False)

    h_xy = (h_xy[0].tolist(), h_xy[1].tolist())
    h_xp = (h_xp[0].tolist(), h_xp[1].tolist())

    # debug / viz
    if debug:
        plt.hist(x_to_y_cos, bins=bins, alpha=0.5)
        plt.hist(x_to_p_cos, bins=bins, alpha=0.5)
        plt.title('distribution of distance to input')
        plt.legend(['Distance from Input to Label',
                    'Distance from Input to Label'],
                   loc='upper right')
        # plt.show()
        plt.savefig(os.path.join(current.report_path,
                                 'plots',
                                 'distance_to_query.png'))
        plt.clf()

    return {
        'histogram_x_to_y': h_xy,
        'histogram_x_to_p': h_xp,
        'raw_distances_x_to_y': x_to_y_cos,
        'raw_distances_x_to_p': x_to_p_cos,
    }

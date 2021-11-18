import numpy as np
from reclist.metrics.standard_metrics import sample_misses_at_k
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

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

    # TODO: Maybe sample some examples from the bins
    histogram = np.histogram(cos_distances, bins=bins, density=False)
    # cast to list
    histogram = (histogram[0].tolist(), histogram[1].tolist())
    # debug / viz
    if debug:
        plt.hist(cos_distances, bins=bins)
        plt.title('dist over cosine distance prod space')
        plt.show()

    return {'mean': np.mean(cos_distances), 'histogram': histogram}


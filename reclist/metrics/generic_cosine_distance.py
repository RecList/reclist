import numpy as np
from reclist.metrics.standard_metrics import sample_misses_at_k, sample_hits_at_k
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt

def generic_cosine_distance(embeddings: dict,
                            type_fn,
                            y_test,
                            y_preds,
                            k=10,
                            bins=25,
                            debug=False):

    misses = sample_misses_at_k(y_preds, y_test, k=k, size=-1)
    cos_distances = []
    for m in misses:
        if m['Y_TEST'] and m['Y_PRED'] and type_fn(m['Y_TEST'][0]) and type_fn(m['Y_PRED'][0]):
            vector_test = embeddings.get(type_fn(m['Y_TEST'][0]), None)
            vector_pred = embeddings.get(type_fn(m['Y_PRED'][0]), None)
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
        plt.title('cosine distance misses')
        plt.show()
    return {'mean': np.mean(cos_distances), 'histogram': histogram}


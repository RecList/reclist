from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
from reclist.current import current
import os
import json
import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path
from statistics import mean
from reclist.metrics.standard_metrics import sample_misses_at_k, sample_hits_at_k
import numpy as np

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



def shortest_path_length():
    pass


get_nodes = lambda nodes, ancestors="": [] if not nodes else ['_'.join([ancestors, nodes[0]])] + \
                                                             get_nodes(nodes[1:], '_'.join([ancestors, nodes[0]]))


def graph_distance_test(y_test, y_preds, product_data, k=3):
    path_lengths = []
    misses = sample_misses_at_k(y_preds, y_test, k=k, size=-1)
    for _y, _y_p in zip([_['Y_TEST'] for _ in misses],
                        [_['Y_PRED'] for _ in misses]):
        if not _y_p:
            continue
        _y_sku = _y[0]
        _y_p_sku = _y_p[0]

        if _y_sku not in product_data or _y_p_sku not in product_data:
            continue
        if not product_data[_y_sku]['CATEGORIES'] or not product_data[_y_p_sku]['CATEGORIES']:
            continue
        # extract graph information
        catA = json.loads(product_data[_y_sku]['CATEGORIES'])
        catB = json.loads(product_data[_y_p_sku]['CATEGORIES'])
        catA_nodes = [get_nodes(c) for c in catA]
        catB_nodes = [get_nodes(c) for c in catB]
        all_nodes = list(set([n for c in catA_nodes + catB_nodes for n in c]))
        all_edges = [(n1, n2) for c in catA_nodes + catB_nodes for n1, n2 in zip(c[:-1], c[1:])]
        all_edges = list(set(all_edges))

        # build graph
        G = nx.Graph()
        G.add_nodes_from(all_nodes)
        G.add_edges_from(all_edges)

        # get leaves
        cat1_leaves = [c[-1] for c in catA_nodes]
        cat2_leaves = [c[-1] for c in catB_nodes]

        all_paths = [shortest_path(G, c1_l, c2_l) for c1_l in cat1_leaves for c2_l in cat2_leaves]
        s_path = min(all_paths, key=len)
        s_path_len = len(s_path) - 1
        path_lengths.append(s_path_len)

    histogram = np.histogram(path_lengths, bins=np.arange(0, max(path_lengths)))
    histogram = (histogram[0].tolist(), histogram[1].tolist())
    return {'mean': mean(path_lengths), 'hist': histogram}



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
        plt.savefig(os.path.join(current.report_path,
                                 'plots',
                                 'cosine_distance_over_type.png'))
        plt.clf()

    return {'mean': np.mean(cos_distances), 'histogram': histogram}

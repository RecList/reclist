import json
import networkx as nx
from networkx.algorithms.shortest_paths.generic import shortest_path
from statistics import mean
from reclist.metrics.standard_metrics import sample_misses_at_k, sample_hits_at_k
import numpy as np


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

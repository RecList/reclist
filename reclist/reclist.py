import collections
import numpy as np
from reclist.abstractions import RecList, rec_test
from typing import List
import random


class CoveoCartRecList(RecList):

    @rec_test(test_type='stats')
    def basic_stats(self):
        """
        Basic statistics on training, test and prediction data
        """
        from reclist.metrics.standard_metrics import statistics
        return statistics(self._x_train,
                          self._y_train,
                          self._x_test,
                          self._y_test,
                          self._y_preds)

    @rec_test(test_type='Coverage@10')
    def coverage_at_k(self):
        """
        Coverage is the proportion of all possible products which the RS
        recommends based on a set of sessions
        """
        from reclist.metrics.standard_metrics import coverage_at_k
        return coverage_at_k(self.sku_only(self._y_preds),
                             self.product_data,
                             k=10)

    @rec_test(test_type='HR@10')
    def hit_rate_at_k(self):
        """
        Compute the rate in which the top-k predictions contain the item to be predicted
        """
        from reclist.metrics.standard_metrics import hit_rate_at_k
        return hit_rate_at_k(self.sku_only(self._y_preds),
                             self.sku_only(self._y_test),
                             k=10)

    @rec_test(test_type='hits_distribution')
    def hits_distribution(self):
        """
        Compute the distribution of hit-rate across product frequency in training data
        """
        from reclist.metrics.hits_distribution import hits_distribution
        return hits_distribution(self.sku_only(self._x_train),
                                 self.sku_only(self._x_test),
                                 self.sku_only(self._y_test),
                                 self.sku_only(self._y_preds),
                                 k=10,
                                 debug=True)

    @rec_test(test_type='distance_to_query')
    def dist_to_query(self):
        """
        Compute the distribution of distance from query to label and query to prediction
        """
        from reclist.metrics.cosine_distance_metrics import distance_to_query
        return distance_to_query(self.rec_model,
                                 self.sku_only(self._x_test),
                                 self.sku_only(self._y_test),
                                 self.sku_only(self._y_preds), k=10, bins=25, debug=True)

    def sku_only(self, l: List[List]):
        return [[e['product_sku'] for e in s] for s in l]


class SpotifySessionRecList(RecList):

    @rec_test(test_type='basic_stats')
    def nep_stats(self):
        """
        Basic statistics on training, test and prediction data for Next Event Prediction
        """
        from reclist.metrics.standard_metrics import statistics
        return statistics(self._x_train,
                          self._y_train,
                          self._x_test,
                          self._y_test,
                          self._y_preds)

    @rec_test(test_type='HR@10')
    def hit_rate_at_k(self):
        """
        Compute the rate at which the top-k predictions contain the item to be predicted
        """
        from reclist.metrics.standard_metrics import hit_rate_at_k
        return hit_rate_at_k(self.uri_only(self._y_preds),
                             self.uri_only(self._y_test),
                             k=10)

    @rec_test(test_type='perturbation_test')
    def perturbation_at_k(self):
        """
        Compute average consistency in model predictions when inputs are perturbed
        """
        from reclist.metrics.perturbation import session_perturbation_test
        from collections import defaultdict
        from functools import partial

        # Step 1: Generate a map from artist uri to track uri
        substitute_mapping = defaultdict(list)
        for track_uri, row in self.product_data.items():
            substitute_mapping[row['artist_uri']].append(track_uri)

        # Step 2: define a custom perturbation function
        def perturb(session, sub_map):
            last_item = session[-1]
            last_item_artist = self.product_data[last_item['track_uri']]['artist_uri']
            substitutes = set(sub_map.get(last_item_artist,[])) - {last_item['track_uri']}
            if substitutes:
                similar_item = random.sample(substitutes, k=1)
                new_session = session[:-1] + [{"track_uri": similar_item[0]}]
                return new_session
            return []

        # Step 3: call test
        return session_perturbation_test(self.rec_model,
                                         self._x_test,
                                         self._y_preds,
                                         partial(perturb, sub_map=substitute_mapping),
                                         self.uri_only,
                                         k=10)

    @rec_test(test_type='hits_distribution_by_playlist_length')
    def hits_distribution_by_slice(self):
        """
        Compute the distribution of hit-rate across various slices of data
        """
        from reclist.metrics.hits_slice import hits_distribution_by_slice

        def generate_slices(x_test, **kwargs):
            slices = collections.defaultdict(list)
            for idx, playlist in enumerate(x_test):
                slices[len(playlist)].append(idx)
            slices = collections.OrderedDict(sorted(slices.items()))
            return slices

        return hits_distribution_by_slice(generate_slices,
                                          self.uri_only(self._x_test),
                                          self.uri_only(self._y_test),
                                          self.uri_only(self._y_preds),
                                          self.product_data,
                                          debug=True)

    @rec_test(test_type='Coverage@10')
    def coverage_at_k(self):
        """
        Coverage is the proportion of all possible products which the RS
        recommends based on a set of sessions
        """
        from reclist.metrics.standard_metrics import coverage_at_k
        return coverage_at_k(self.uri_only(self._y_preds),
                             self.product_data,
                             # this contains all the track URIs from train and test sets
                             k=10)

    @rec_test(test_type='Popularity@10')
    def popularity_bias_at_k(self):
        """
        Compute average frequency of occurrence across recommended items in training data
        """
        from reclist.metrics.standard_metrics import popularity_bias_at_k
        return popularity_bias_at_k(self.uri_only(self._y_preds),
                                    self.uri_only(self._x_train),
                                    k=10)

    @rec_test(test_type='MRR@10')
    def mrr_at_k(self):
        """
        MRR calculates the mean reciprocal of the rank at which the first
        relevant item was retrieved
        """
        from reclist.metrics.standard_metrics import mrr_at_k
        return mrr_at_k(self.uri_only(self._y_preds),
                        self.uri_only(self._y_test))

    def uri_only(self, playlists: List[dict]):
        return [[track['track_uri'] for track in playlist] for playlist in playlists]

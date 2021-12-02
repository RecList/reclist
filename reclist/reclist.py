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
        return distance_to_query(self.rec_model,]8
        u
                                 self.sku_only(self._x_test),
                                 self.sku_only(self._y_test),
                                 self.sku_only(self._y_preds), k=10, bins=25, debug=True)

    def sku_only(self, l: List[List]):
        return [[e['product_sku'] for e in s] for s in l]


class SpotifySessionRecList(RecList):

    ########### NEXT EVENT PREDICTION #########
    @rec_test(test_type='NEP_stats')
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

    @rec_test(test_type='NEP_HR@10')
    def hit_rate_at_k(self):
        """
        Compute the rate at which the top-k predictions contain the item to be predicted
        """
        from reclist.metrics.standard_metrics import hit_rate_at_k
        return hit_rate_at_k(self.uri_only(self._y_preds),
                             self.uri_only(self._y_test),
                             k=10)

    # @rec_test(test_type='NEP_perturbation')
    # def perturbation_at_k(self):
    #     """
    #     Compute average consistency in model predictions when inputs are perturbed
    #     """
    #     from reclist.metrics.perturbation import session_perturbation_test
    #     from collections import defaultdict
    #     from functools import partial
    #
    #     x_test, y_test = self.generate_nep_test_set()
    #     y_preds = self.get_y_preds(x_test, y_test)
    #
    #     # map from artist uri to track uri
    #     substitute_mapping = defaultdict(list)
    #     for track_uri, row in self.product_data.items():
    #         substitute_mapping[row['artist_uri']].append(track_uri)
    #
    #     # define a custom perturbation function
    #     def perturb(session, sub_map):
    #         last_item = session[-1]
    #         last_item_artist = self.product_data[last_item['track_uri']]['artist_uri']
    #         substitutes = set(sub_map.get(last_item_artist,[])) - {last_item['track_uri']}
    #         if substitutes:
    #             similar_item = random.sample(substitutes, k=1)
    #             new_session = session[:-1] + [{"track_uri": similar_item[0]}]
    #             return new_session
    #         return []
    #
    #     # call test
    #     return session_perturbation_test(self.rec_model,
    #                                      x_test,
    #                                      y_preds,
    #                                      partial(perturb, sub_map=substitute_mapping),
    #                                      self.uri_only,
    #                                      k=10)
    #
    #
    # @rec_test(test_type='NEP_hits_distribution_by_slice')
    # def hits_distribution_by_slice(self):
    #     """
    #     Compute the distribution of hit-rate across various slices of data
    #     """
    #     from reclist.metrics.hits_slice import hits_distribution_by_slice
    #     x_test, y_test = self.generate_nep_test_set()
    #     y_preds = self.get_y_preds(x_test, y_test)
    #
    #     # slice by artist
    #     slice_fns = {
    #         'HIP-HOP/RAP': lambda _: _['artist_uri'] == '3TVXtAsR1Inumwj472S9r4',  # Drake
    #         'POP': lambda _: _['artist_uri'] == '5pKCCKE2ajJHZ9KAiaK11H',  # Rihanna
    #         'EDM': lambda _: _['artist_uri'] == '69GGBxA162lTqCwzJG5jLp',  # The Chainsmokers
    #         'R&B': lambda _: _['artist_uri'] == '1Xyo4u8uXC1ZmMpatF05PJ',  # The Weeknd
    #     }
    #
    #     return hits_distribution_by_slice(slice_fns,
    #                                       self.uri_only(y_test),
    #                                       self.uri_only(y_preds),
    #                                       self.product_data,
    #                                       debug=True)
    #
    # @rec_test(test_type='NEP_Coverage@10')
    # def coverage_at_k(self):
    #     """
    #     Coverage is the proportion of all possible products which the RS
    #     recommends based on a set of sessions
    #     """
    #     from reclist.metrics.standard_metrics import coverage_at_k
    #     x_test, y_test = self.generate_nep_test_set()
    #     y_preds = self.get_y_preds(x_test, y_test)
    #     return coverage_at_k(self.uri_only(y_preds),
    #                          self.product_data,
    #                          # this contains all the track URIs from train and test sets
    #                          k=10)
    #
    # @rec_test(test_type='Popularity@10')
    # def popularity_bias_at_k(self):
    #     """
    #     Compute average frequency of occurrence across recommended items in training data
    #     """
    #     from reclist.metrics.standard_metrics import popularity_bias_at_k
    #     x_test, y_test = self.generate_nep_test_set()
    #     y_preds = self.get_y_preds(x_test, y_test)
    #     return popularity_bias_at_k(self.uri_only(y_preds),
    #                                 self.uri_only(self._x_train),
    #                                 k=10)
    #
    # @rec_test(test_type='MRR@10')
    # def mrr_at_k(self):
    #     """
    #     MRR calculates the mean reciprocal of the rank at which the first
    #     relevant item was retrieved
    #     """
    #     from reclist.metrics.standard_metrics import mrr_at_k
    #     x_test, y_test = self.generate_nep_test_set()
    #     y_preds = self.get_y_preds(x_test, y_test)
    #     return mrr_at_k(self.uri_only(y_preds),
    #                     self.uri_only(y_test),
    #                     k=10)
    #
    #
    # @rec_test(test_type='NEP_perturbation')
    # def perturbation_at_k(self):
    #     """
    #     Compute average consistency in model predictions when inputs are perturbed
    #     """
    #     from reclist.metrics.perturbation import session_perturbation_test
    #     from collections import defaultdict
    #     from functools import partial
    #
    #     x_test, y_test = self.generate_nep_test_set()
    #     y_preds = self.get_y_preds(x_test, y_test)
    #
    #     # should we use the whole catalog or just the ones present in the training set?
    #     catalog = collections.defaultdict(dict)
    #     for dataset in [self._x_train]:
    #         for playlist in dataset:
    #             for track in playlist['tracks']:
    #                 if track['track_uri'] in catalog:
    #                     continue  # could also double check that the existing info lines up
    #                 catalog[track['track_uri']] = {
    #                     'artist_uri': track['artist_uri'],
    #                     'album_uri': track['album_uri'],
    #                     'duration_ms': track['duration_ms']
    #                 }
    #     # map from artist uri to track uri
    #     substitute_mapping = defaultdict(list)
    #     for track_uri, row in catalog.items():
    #         substitute_mapping[row['artist_uri']].append(track_uri)
    #
    #     # define a custom perturbation function
    #     def perturb(session, sub_map):
    #         last_item = session[-1]
    #         last_item_artist = last_item['artist_uri']
    #         substitutes = set(sub_map.get(last_item_artist,[])) - {last_item['track_uri']}
    #         if substitutes:
    #             similar_item = random.sample(substitutes, k=1)
    #             new_session = session[:-1] + [{"track_uri": similar_item[0]}]
    #             return new_session
    #         return []
    #
    #     # call test
    #     return session_perturbation_test(self.rec_model,
    #                                      x_test,
    #                                      y_preds,
    #                                      partial(perturb, sub_map=substitute_mapping),
    #                                      self.uri_only,
    #                                      k=10)

    # ########### ALL SUBSEQUENT PREDICTION #########
    # @rec_test(test_type='ALL_stats')
    # def all_subsequent_stats(self):
    #     """
    #     Basic statistics on training, test and prediction data for all subsequent prediction
    #     """
    #     from reclist.metrics.standard_metrics import statistics
    #     x_test, y_test = self.generate_all_subsequent_test_set()
    #     y_preds = self.get_y_preds(x_test, y_test, overwrite=True)
    #     return statistics(self._x_train,
    #                       self._y_train,
    #                       x_test,
    #                       y_test,
    #                       y_preds)
    #
    # @rec_test(test_type='ALL_P@50')
    # def precision_at_k(self):
    #     """
    #     Compute the proportion of recommended items in the top-k set that are relevant
    #     """
    #     from reclist.metrics.standard_metrics import precision_at_k
    #     x_test, y_test = self.generate_all_subsequent_test_set()
    #     y_preds = self.get_y_preds(x_test, y_test)
    #     return precision_at_k(self.uri_only(y_preds),
    #                           self.uri_only(y_test),
    #                           k=50)
    #
    # @rec_test(test_type='ALL_R@50')
    # def recall_at_k(self):
    #     """
    #     Compute the proportion of relevant items found in the top-k recommendations
    #     """
    #     from reclist.metrics.standard_metrics import recall_at_k
    #     x_test, y_test = self.generate_all_subsequent_test_set()
    #     y_preds = self.get_y_preds(x_test, y_test)
    #     return recall_at_k(self.uri_only(y_preds),
    #                        self.uri_only(y_test),
    #                        k=50)
    #
    # @rec_test(test_type='ALL_MRR@10')
    # def mrr_at_k(self):
    #     """
    #     MRR calculates the mean reciprocal of the rank at which the first
    #     relevant item was retrieved
    #     """
    #     from reclist.metrics.standard_metrics import mrr_at_k
    #     x_test, y_test = self.generate_all_subsequent_test_set()
    #     y_preds = self.get_y_preds(x_test, y_test)
    #     return mrr_at_k(self.uri_only(y_preds),
    #                     self.uri_only(y_test),
    #                     k=10)

    def uri_only(self, playlists: List[dict]):
        return [[track['track_uri'] for track in playlist] for playlist in playlists]

    def generate_nep_test_set(
        self,
        shuffle: bool = True,
        seed: int = 0
    ):
        """
        Generate test set for Next Event Prediction (NEP) (i.e. predict the immediate
        next item given the first n items) by taking last item as target
        """
        x_test = []
        y_test = []
        for playlist in self._x_test:
            if len(playlist) < 2:
                continue
            x_test.append(playlist[:-1])
            y_test.append([playlist[-1]])

        if shuffle:
            random.seed(seed)
            test = list(zip(x_test, y_test))
            random.shuffle(test)
            x_test, y_test = zip(*test)

        return x_test, y_test

    def generate_all_subsequent_test_set(
        self,
        k: int = 5,
        shuffle: bool = False,
        seed: int = 0
    ):
        """
        Generate test set for all subsequent prediction, i.e. hold out k items as seed,
        use the rest of the items as ground truth.
        """
        x_test = []
        y_test = []
        for playlist in self._x_test:
            if len(playlist) <= k:
                continue
            if shuffle:
                random.seed(seed)
                all_idx = list(range(len(playlist)))
                seeded_idx = random.sample(all_idx, k=k)
                held_out_idx = set(all_idx) - set(seeded_idx)
                seeded_tracks = [playlist[idx] for idx in seeded_idx]
                held_out_tracks = [playlist[idx] for idx in held_out_idx]
                assert len(seeded_tracks) + \
                       len(held_out_tracks) == len(playlist)
            else:
                seeded_tracks = playlist[:k]
                held_out_tracks = playlist[k:]
            x_test.append(seeded_tracks)
            y_test.append(held_out_tracks)

        return x_test, y_test

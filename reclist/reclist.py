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
        y_preds = self.rec_model.predict(self._x_test)
        return statistics(self._x_train,
                          self._y_train,
                          self._x_test,
                          self._y_test,
                          y_preds)

    @rec_test(test_type='Coverage@10')
    def coverage_at_k(self):
        """
        Coverage is the proportion of all possible products which the RS
        recommends based on a set of sessions
        """
        from reclist.metrics.standard_metrics import coverage_at_k
        y_preds = self.rec_model.predict(self._x_test)
        return coverage_at_k(self.sku_only(y_preds),
                             self.product_data,
                             k=10)

    @rec_test(test_type='HR@10')
    def hit_rate_at_k(self):
        """
        Compute the rate in which the top-k predictions contain the item to be predicted
        """
        y_preds = self.rec_model.predict(self._x_test)
        from reclist.metrics.standard_metrics import hit_rate_at_k
        return hit_rate_at_k(self.sku_only(y_preds),
                             self.sku_only(self._y_test),
                             k=10)

    @rec_test(test_type='hits_distribution')
    def hits_distribution(self):
        """
        Compute the distribution of hit-rate across product frequency in training data
        """
        from reclist.metrics.hits_distribution import hits_distribution
        y_preds = self.rec_model.predict(self._x_test)

        return hits_distribution(self.sku_only(self._x_train),
                                 self.sku_only(self._x_test),
                                 self.sku_only(self._y_test),
                                 self.sku_only(y_preds),
                                 k=10,
                                 debug=True)

    @rec_test(test_type='distance_to_query')
    def dist_to_query(self):
        """
        Compute the distribution of distance from query to label and query to prediction
        """
        y_preds = self.rec_model.predict(self._x_test)
        from reclist.metrics.cosine_distance_metrics import distance_to_query
        return distance_to_query(self.rec_model,
                                 self.sku_only(self._x_test),
                                 self.sku_only(self._y_test),
                                 self.sku_only(y_preds), k=10, bins=25, debug=True)

    def sku_only(self, l:List[List]):
        return [[e['product_sku'] for e in s] for s in l]


class SpotifySessionRecList(RecList):

    ########### NEXT EVENT PREDICTION #########
    @rec_test(test_type='NEP_stats')
    def nep_stats(self):
        """
        Basic statistics on training, test and prediction data for Next Event Prediction
        """
        from reclist.metrics.standard_metrics import statistics
        x_test, y_test = self.generate_nep_test_set()
        y_preds = self.get_y_preds(x_test, y_test, overwrite=True)
        return statistics(self._x_train,
                          self._y_train,
                          x_test,
                          y_test,
                          [playlist['tracks'] for playlist in y_preds])

    @rec_test(test_type='NEP_HR@10')
    def hit_rate_at_k(self):
        """
        Compute the rate at which the top-k predictions contain the item to be predicted
        """
        from reclist.metrics.standard_metrics import hit_rate_at_k
        x_test, y_test = self.generate_nep_test_set()
        y_preds = self.get_y_preds(x_test, y_test)
        return hit_rate_at_k(self.uri_only(y_preds),
                             self.uri_only(y_test),
                             k=10)

    # @rec_test(test_type='NEP_hits_distribution')
    # def hits_distribution(self):
    #     """
    #     Compute the distribution of hit-rate across product frequency in training data
    #     """
    #     from reclist.metrics.hits_distribution import hits_distribution
    #     x_test, y_test = self.generate_nep_test_set()
    #     y_preds = self.get_y_preds(x_test, y_test)
    #     return hits_distribution(self.uri_only(self._x_train),
    #                              self.uri_only(self._x_test),
    #                              self.uri_only(y_test),
    #                              self.uri_only(y_preds),
    #                              k=10,
    #                              debug=True)

    @rec_test(test_type='NEP_perturbation')
    def perturbation_at_k(self):
        """
        Compute average consistency in model predictions when inputs are perturbed
        """
        # from reclist.metrics.perturbation import session_perturbation_test
        x_test, y_test = self.generate_nep_test_set()
        y_preds = self.get_y_preds(x_test, y_test)

        # should we use the whole catalog or just the ones present in the training set?
        catalog = collections.defaultdict(dict)
        for dataset in [self._x_train, x_test, y_test]:
            for playlist in dataset:
                for track in playlist['tracks']:
                    if track['track_uri'] in catalog:
                        continue  # could also double check that the existing info lines up
                    catalog[track['track_uri']]= {
                        'artist_uri': track['artist_uri'],
                        'album_uri': track['album_uri'],
                        'duration_ms': track['duration_ms']
                    }

        def get_item_with_category(product_data: dict, category: set, to_ignore=None):
            to_ignore = [] if to_ignore is None else to_ignore
            uris = [_ for _ in product_data if product_data[_]['artist_uri'] == category and _ not in to_ignore]  # this is a really expensive operation
            if uris:
                return random.choice(uris)
            return []

        def perturb_session(session, product_data):
            last_item = session[-1]
            last_item_category = last_item['artist_uri']
            similar_item = get_item_with_category(product_data, last_item_category, to_ignore=[last_item])
            if similar_item:
                new_session = session[:-1] + [{"track_uri": similar_item}]
                return new_session
            return []

        def session_perturbation_test(model, x_test, y_preds, product_data, k=None):
            overlap_ratios = []
            y_p = []
            s_perturbs = []

            # generate a batch of perturbations
            for s, _y_p in zip(x_test, y_preds):
                # perturb last item in session
                s_perturb = perturb_session(s['tracks'], product_data)
                if not s_perturb:
                    continue
                s_perturbs.append({'tracks': s_perturb})
                y_p.append(_y_p)

            y_perturbs = model.predict(s_perturbs)

            y_p, y_perturbs = self.uri_only(y_p), self.uri_only(y_perturbs)
            if k is None:
                k = min(len(y_p[0]), len(y_perturbs[0]))

            for _y_p, _y_perturb in zip(y_p, y_perturbs):
                if _y_p and _y_perturb:
                    # compute prediction intersection
                    intersection = set(_y_perturb[:k]).intersection(_y_p[:k])
                    overlap_ratio = len(intersection)/len(_y_p[:k])
                    overlap_ratios.append(overlap_ratio)

            return np.mean(overlap_ratios)

        return session_perturbation_test(self.rec_model,
                                         x_test,
                                         y_preds,
                                         catalog)

    # @rec_test(test_type='NEP_hits_distribution_by_artist_popularity_slice')
    # def hits_distribution_by_artist_popularity_slice(self):
    #     """
    #     Compute the distribution of hit-rate across various slices of data
    #     based on artist popularity
    #     """
    #     # from reclist.metrics.hits_slice import hits_distribution_by_slice
    #     x_test, y_test = self.generate_nep_test_set()
    #     y_preds = self.get_y_preds(x_test, y_test)

    #     # create catalog with metadata that will be used for slicinng
    #     catalog = collections.defaultdict(dict)
    #     for dataset in [self._x_train, x_test, y_test]:
    #         for playlist in dataset:
    #             for track in playlist['tracks']:
    #                 if track['track_uri'] in catalog:
    #                     continue  # could also double check that the existing info lines up
    #                 catalog[track['track_uri']]= {
    #                     'artist_uri': track['artist_uri'],
    #                     'album_uri': track['album_uri'],
    #                     'duration_ms': track['duration_ms']
    #                 }

    #     # top 10 and bottom 10 artists by popularity
    #     artist_popularity = collections.Counter()
    #     for playlist in self._x_train:
    #         for track in playlist['tracks']:
    #             artist_popularity[track['artist_uri']] += 1
    #     # there are a lot of ties among not-so-popular artists, so we filter
    #     # out the really uncommon ones
    #     artist_popularity = collections.Counter({k: v for k, v in artist_popularity.items() if v >= 100})
    #     top_artists = [a for a, _ in artist_popularity.most_common()[:10]]
    #     bottom_artists = [a for a, _ in artist_popularity.most_common()[::-1][:50]]

    #     slice_fns = {f'TOP_{i}': lambda _, a=a: _['artist_uri'] == a for i, a \
    #         in enumerate(top_artists, start=1)}
    #     slice_fns.update({f'BOTTOM_{i}': lambda _, a=a: _['artist_uri'] == a for i, a \
    #         in enumerate(bottom_artists, start=1)})

    #     def hits_distribution_by_slice(slice_fns: dict,
    #                                    y_test,
    #                                    y_preds,
    #                                    product_data,
    #                                    k=3,
    #                                    sample_size=3,
    #                                    format_fn=None,
    #                                    debug=False):

    #         from reclist.metrics.standard_metrics import hit_rate_at_k
    #         import matplotlib.pyplot as plt

    #         hit_rate_per_slice = collections.defaultdict(dict)
    #         for slice_name, filter_fn in slice_fns.items():
    #             # get indices for slice
    #             slice_idx = [idx for idx, _y in enumerate(y_test) if _y['tracks'][0]['track_uri'] \
    #                 in product_data and filter_fn(product_data[_y['tracks'][0]['track_uri']])]
    #             if not slice_idx:
    #                 continue
    #             # get predictions for slice
    #             slice_y_preds = [y_preds[_] for _ in slice_idx]
    #             # get labels for slice
    #             slice_y_test = [y_test[_] for _ in slice_idx]
    #             if format_fn:
    #                 slice_y_preds, slice_y_test = format_fn(slice_y_preds), format_fn(slice_y_test)
    #             # TODO: We may want to allow for generic metric to be used here
    #             slice_hr = hit_rate_at_k(slice_y_preds, slice_y_test, k=k)
    #             # store results
    #             hit_rate_per_slice[slice_name]['hit_rate'] = slice_hr
    #             # TODO: Need to fix sample_hits_at_k and sample_misses_at_k for NEP vs ALL
    #             # hit_rate_per_slice[slice_name]['hits'] = sample_hits_at_k(slice_y_preds, slice_y_test, k=k, size=sample_size)
    #             # hit_rate_per_slice[slice_name]['misses'] = sample_misses_at_k(slice_y_preds, slice_y_test, k=k, size=sample_size)

    #         # debug / visualization
    #         if debug:
    #             x_tick_names = list(hit_rate_per_slice.keys())
    #             x_tick_idx = list(range(len(x_tick_names)))
    #             plt.bar(x_tick_idx, [v['hit_rate'] for v in hit_rate_per_slice.values()], align='center')
    #             plt.xticks(list(range(len(hit_rate_per_slice))), x_tick_names)
    #             plt.show()

    #         # cast to normal dict
    #         return dict(hit_rate_per_slice)

    #     return hits_distribution_by_slice(slice_fns,
    #                                       y_test,
    #                                       y_preds,
    #                                       catalog,
    #                                       format_fn=self.uri_only,
    #                                       debug=True)

    @rec_test(test_type='NEP_hits_distribution_by_slice')
    def hits_distribution_by_slice(self):
        """
        Compute the distribution of hit-rate across various slices of data
        """
        # from reclist.metrics.hits_slice import hits_distribution_by_slice
        x_test, y_test = self.generate_nep_test_set()
        y_preds = self.get_y_preds(x_test, y_test)

        # create catalog with metadata that will be used for slicinng
        catalog = collections.defaultdict(dict)
        for dataset in [self._x_train, x_test, y_test]:
            for playlist in dataset:
                for track in playlist['tracks']:
                    if track['track_uri'] in catalog:
                        continue  # could also double check that the existing info lines up
                    catalog[track['track_uri']]= {
                        'artist_uri': track['artist_uri'],
                        'album_uri': track['album_uri'],
                        'duration_ms': track['duration_ms']
                    }

        # import matplotlib.pyplot as plt
        # (n, bins, patches) = plt.hist([t['duration_ms'] for t in catalog.values()], bins=50)
        # print(n)
        # plt.show()

        # genre
        slice_fns = {
            'HIP-HOP/RAP': lambda _: _['artist_uri'] == '3TVXtAsR1Inumwj472S9r4',  # Drake
            'POP': lambda _: _['artist_uri'] == '5pKCCKE2ajJHZ9KAiaK11H',  # Rihanna
            'EDM': lambda _: _['artist_uri'] == '69GGBxA162lTqCwzJG5jLp',  # The Chainsmokers
            'R&B': lambda _: _['artist_uri'] == '1Xyo4u8uXC1ZmMpatF05PJ',  # The Weeknd
        }

        def hits_distribution_by_slice(slice_fns: dict,
                                       y_test,
                                       y_preds,
                                       product_data,
                                       k=3,
                                       sample_size=3,
                                       format_fn=None,
                                       debug=False):

            from reclist.metrics.standard_metrics import hit_rate_at_k
            import matplotlib.pyplot as plt

            hit_rate_per_slice = collections.defaultdict(dict)
            for slice_name, filter_fn in slice_fns.items():
                # get indices for slice
                slice_idx = [idx for idx, _y in enumerate(y_test) if _y['tracks'][0]['track_uri'] \
                    in product_data and filter_fn(product_data[_y['tracks'][0]['track_uri']])]
                if not slice_idx:
                    continue
                # get predictions for slice
                slice_y_preds = [y_preds[_] for _ in slice_idx]
                # get labels for slice
                slice_y_test = [y_test[_] for _ in slice_idx]
                if format_fn:
                    slice_y_preds, slice_y_test = format_fn(slice_y_preds), format_fn(slice_y_test)
                # TODO: We may want to allow for generic metric to be used here
                slice_hr = hit_rate_at_k(slice_y_preds, slice_y_test, k=k)
                # store results
                hit_rate_per_slice[slice_name]['hit_rate'] = slice_hr
                # TODO: Need to fix sample_hits_at_k and sample_misses_at_k for NEP vs ALL
                # hit_rate_per_slice[slice_name]['hits'] = sample_hits_at_k(slice_y_preds, slice_y_test, k=k, size=sample_size)
                # hit_rate_per_slice[slice_name]['misses'] = sample_misses_at_k(slice_y_preds, slice_y_test, k=k, size=sample_size)

            # debug / visualization
            if debug:
                x_tick_names = list(hit_rate_per_slice.keys())
                x_tick_idx = list(range(len(x_tick_names)))
                plt.bar(x_tick_idx, [v['hit_rate'] for v in hit_rate_per_slice.values()], align='center')
                plt.xticks(list(range(len(hit_rate_per_slice))), x_tick_names)
                plt.show()

            # cast to normal dict
            return dict(hit_rate_per_slice)

        return hits_distribution_by_slice(slice_fns,
                                          y_test,
                                          y_preds,
                                          catalog,
                                          format_fn=self.uri_only,
                                          debug=True)

    @rec_test(test_type='NEP_Coverage@10')
    def coverage_at_k(self):
        """
        Coverage is the proportion of all possible products which the RS
        recommends based on a set of sessions
        """
        from reclist.metrics.standard_metrics import coverage_at_k
        x_test, y_test = self.generate_nep_test_set()
        y_preds = self.get_y_preds(x_test, y_test)
        return coverage_at_k(self.uri_only(y_preds),
                             self.product_data['uri2track'],  # this contains all the track URIs from train and test sets
                             k=10)

    @rec_test(test_type='NEP_Popularity@10')
    def popularity_bias_at_k(self):
        """
        Compute average frequency of occurrence across recommended items in training data
        """
        from reclist.metrics.standard_metrics import popularity_bias_at_k
        x_test, y_test = self.generate_nep_test_set()
        y_preds = self.get_y_preds(x_test, y_test)
        return popularity_bias_at_k(self.uri_only(y_preds),
                                    self.uri_only(self._x_train),
                                    k=10)

    ########### ALL SUBSEQUENT PREDICTION #########
    @rec_test(test_type='ALL_stats')
    def all_subsequent_stats(self):
        """
        Basic statistics on training, test and prediction data for all subsequent prediction
        """
        from reclist.metrics.standard_metrics import statistics
        x_test, y_test = self.generate_all_subsequent_test_set()
        y_preds = self.get_y_preds(x_test, y_test, overwrite=True)
        return statistics(self._x_train,
                          self._y_train,
                          x_test,
                          y_test,
                          [playlist['tracks'] for playlist in y_preds])

    @rec_test(test_type='ALL_P@50')
    def precision_at_k(self):
        """
        Compute the proportion of recommended items in the top-k set that are relevant
        """
        from reclist.metrics.standard_metrics import precision_at_k
        x_test, y_test = self.generate_all_subsequent_test_set()
        y_preds = self.get_y_preds(x_test, y_test)
        return precision_at_k(self.uri_only(y_preds),
                              self.uri_only(y_test),
                              k=50)

    @rec_test(test_type='ALL_R@50')
    def recall_at_k(self):
        """
        Compute the proportion of relevant items found in the top-k recommendations
        """
        from reclist.metrics.standard_metrics import recall_at_k
        x_test, y_test = self.generate_all_subsequent_test_set()
        y_preds = self.get_y_preds(x_test, y_test)
        return recall_at_k(self.uri_only(y_preds),
                           self.uri_only(y_test),
                           k=50)

    @rec_test(test_type='ALL_MRR@10')
    def mrr_at_k(self):
        """
        MRR calculates the mean reciprocal of the rank at which the first
        relevant item was retrieved
        """
        from reclist.metrics.standard_metrics import mrr_at_k
        x_test, y_test = self.generate_all_subsequent_test_set()
        y_preds = self.get_y_preds(x_test, y_test)
        return mrr_at_k(self.uri_only(y_preds),
                        self.uri_only(y_test),
                        k=10)

    def uri_only(self, playlists: List[dict]):
        return [[track['track_uri'] for track in playlist['tracks']] for playlist in playlists]

    def generate_nep_test_set(
        self,
        n: int = 1,
        iteratively_increment: bool = True,
        shuffle: bool = True,
        seed: int = 0
    ):
        """
        Generate test set for Next Event Prediction (NEP), i.e. predict the immediate
        next item given the first n items.
        """
        x_test = []
        y_test = []
        for playlist in self._x_test:
            num_items_given = list(range(n, len(playlist['tracks']), 1)) if iteratively_increment else [n]
            for _n in num_items_given:
                seeded_tracks = playlist['tracks'][:_n]
                next_track = playlist['tracks'][_n]
                x_test.append({
                    'pid': playlist['pid'],
                    'tracks': seeded_tracks
                })
                y_test.append({
                    'pid': playlist['pid'],
                    'tracks': [next_track]
                })

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
            if len(playlist['tracks']) <= k:
                continue
            if shuffle:
                random.seed(seed)
                all_idx = list(range(len(playlist['tracks'])))
                seeded_idx = random.sample(all_idx, k=k)
                held_out_idx = set(all_idx) - set(seeded_idx)
                seeded_tracks = [playlist['tracks'][idx] for idx in seeded_idx]
                held_out_tracks = [playlist['tracks'][idx]
                                   for idx in held_out_idx]
                assert len(seeded_tracks) + \
                    len(held_out_tracks) == len(playlist['tracks'])
            else:
                seeded_tracks = playlist['tracks'][:k]
                held_out_tracks = playlist['tracks'][k:]
            x_test.append({
                'pid': playlist['pid'],
                'tracks': seeded_tracks
            })
            y_test.append({
                'pid': playlist['pid'],
                'tracks': held_out_tracks
            })

        return x_test, y_test

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

    @rec_test(test_type='HR@10')
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

    @rec_test(test_type='hits_distribution')
    def hits_distribution(self):
        """
        Compute the distribution of hit-rate across product frequency in training data
        """
        from reclist.metrics.hits_distribution import hits_distribution
        x_test, y_test = self.generate_nep_test_set()
        y_preds = self.get_y_preds(x_test, y_test)
        return hits_distribution(self.uri_only(self._x_train),
                                 self.uri_only(self._x_test),
                                 self.uri_only(y_test),
                                 self.uri_only(y_preds),
                                 k=10,
                                 debug=True)

    ########### ALL SUBSEQUENT PREDICTION #########
    @rec_test(test_type='all_subsequent_stats')
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

    @rec_test(test_type='P@50')
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
    
    @rec_test(test_type='R@50')
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
            num_items_given = list(range(n, len(playlist) - 1, 1)) if iteratively_increment else [n]
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
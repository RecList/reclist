from reclist.abstractions import RecList, rec_test
from typing import List

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

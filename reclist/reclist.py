from reclist.abstractions import RecList, rec_test
from typing import List

class CoveoCartRecList(RecList):

    @rec_test(test_type='stats')
    def basic_stats(self):
        from reclist.metrics.standard_metrics import statistics
        return statistics(self._x_train,
                          self._y_train,
                          self._x_test,
                          self._y_test,
                          self._y_preds)

    @rec_test(test_type='coverage')
    def coverage_at_k(self):
        """
        Coverage is the proportion of all possible products which the RS
        recommends based on a set of sessions
        """
        from reclist.metrics.standard_metrics import coverage_at_k
        return coverage_at_k(self.sku_only(self._y_preds), self.product_data, k=3)

    @rec_test(test_type='HR@l')
    def hit_rate_at_k(self):
        """
        Computes the rate in which the top-k predictions contain the item to be predicted
        """
        from reclist.metrics.standard_metrics import hit_rate_at_k
        return hit_rate_at_k(self.sku_only(self._y_preds), self.sku_only(self._y_test), k=3)

    @rec_test(test_type='hits_distribution')
    def hits_distribution(self):
        """
        Computes the distribution of hit-rate across product frequency in training data
        """
        from reclist.metrics.hits_distribution import hits_distribution
        return hits_distribution(self.sku_only(self._x_train),
                                 self.sku_only(self._x_test),
                                 self.sku_only(self._y_test),
                                 self.sku_only(self._y_preds),
                                 debug=True)

    @rec_test(test_type='cosine_distance')
    def cosine_distance(self):
        """
        Computes the average cosine distance for missed predictions
        """
        from reclist.metrics.error_by_cosine_distance import error_by_cosine_distance

        result = error_by_cosine_distance(self.rec_model,
                                         self.sku_only(self._y_test),
                                         self.sku_only(self._y_preds),
                                         debug=True)
        return result

    def sku_only(self, l:List[List]):
        return [[e['product_sku'] for e in s] for s in l]

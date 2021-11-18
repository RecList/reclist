from reclist.reclist import RecList, rec_test
from typing import List

class CoveoCartRecList(RecList):

    @rec_test(rec_type='session', test_type='stats')
    def basic_stats(self):
        from reclist.rectest.standard_metrics import statistics
        return statistics(self._x_train,
                          self._y_train,
                          self._x_test,
                          self._y_test,
                          self._y_preds)

    @rec_test(rec_type='session', test_type='coverage')
    def coverage_at_k(self):
        """
        Coverage is the proportion of all possible products which the RS
        recommends based on a set of sessions
        """
        from reclist.rectest.standard_metrics import coverage_at_k
        return coverage_at_k(self.sku_only(self._y_preds), self.product_data, k=3)

    @rec_test(rec_type='session', test_type='HR@10')
    def hit_rate_at_k(self):
        """
        Computes the rate in which the top-k predictions contain the item to be predicted
        """
        from reclist.rectest.standard_metrics import hit_rate_at_k
        return hit_rate_at_k(self.sku_only(self._y_preds), self.sku_only(self._y_test), k=3)

    @rec_test(rec_type='session', test_type='hits_distribution')
    def hits_distribution(self):
        """
        Computes the distribution of hit-rate across product frequency in training data
        """
        from reclist.rectest.hits_distribution import hits_distribution
        return hits_distribution(self.sku_only(self._x_train),
                                 self.sku_only(self._x_test),
                                 self.sku_only(self._y_test),
                                 self.sku_only(self._y_preds),
                                 debug=True)

    @rec_test(rec_type='cart', test_type='category_distance')
    def cat_distance(self):
        """
        Measures the graph distance between prediction and ground truth
        """
        from reclist.rectest.graph_distance_test import graph_distance_test

        return graph_distance_test(self.sku_only(self._y_test),
                                   self.sku_only(self._y_preds),
                                   self.product_data)

    @rec_test(rec_type='session', test_type='brand_cosine_distance')
    def brand_cosine_distance(self):
        from reclist.rectest.generic_cosine_distance import generic_cosine_distance
        # function to return property/type of a product
        def type_fn(event):
            if event['product_sku'] in self.product_data:
                return self.product_data[event['product_sku']]['STANDARDIZED_BRAND']
            else:
                return None

        # train a dense space with word2vec
        self.train_dense_repr('BRAND', type_fn)

        return generic_cosine_distance(embeddings=self._dense_repr['BRAND'],
                                       type_fn=type_fn,
                                       y_test=self._y_test,
                                       y_preds=self._y_preds,
                                       debug=True)

    @rec_test(rec_type='session', test_type='cosine_distance')
    def cosine_distance(self):
        """
        Computes the average cosine distance for missed predictions
        """
        from reclist.rectest.error_by_cosine_distance import error_by_cosine_distance

        result = error_by_cosine_distance(self.rec_model,
                                         self.sku_only(self._y_test),
                                         self.sku_only(self._y_preds),
                                         debug=True)
        return result

    @rec_test(rec_type='session', test_type='price_homogeneity')
    def price_homogeneity(self):
        """
        Computes the average absolute log ratio of predicted item and item in cart
        """
        from reclist.rectest.price_homogeneity import price_homogeneity_test

        return price_homogeneity_test(self.sku_only(self._y_test),
                                      self.sku_only(self._y_preds),
                                      self.product_data)

    @rec_test(rec_type='session', test_type='hits_slices')
    def hits_slices(self):
        """
        Computes the distribution of hit-rate across various sliceS of data in training data
        """
        from reclist.rectest.hits_slice import hits_distribution_by_slice

        slice_fns = {
            'nike': lambda _: _['STANDARDIZED_BRAND'] == 'nike',
            'adidas': lambda _: _['STANDARDIZED_BRAND'] == 'adidas',
            'asics': lambda _: _['STANDARDIZED_BRAND'] == 'asics',
            'puma': lambda _: _['STANDARDIZED_BRAND'] == 'puma',
            'under armour': lambda _: _['STANDARDIZED_BRAND'] == 'under armour',
        }

        return hits_distribution_by_slice(slice_fns,
                                          self.sku_only(self._y_test),
                                          self.sku_only(self._y_preds),
                                          self.product_data)

    def sku_only(self, l:List[List]):
        return [[e['product_sku'] for e in s] for s in l]

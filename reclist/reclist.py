import collections
import random
from typing import List

import pandas as pd

from reclist.abstractions import RecList, plot_test_model, rec_test


class MovieLensRecList(RecList):
    def squared_error(self, y_true, y_pred):
        return (y_true - y_pred) ** 2

    def rmse_by_slice(self, slice: str):
        sq_error = self.squared_error(
            self._y_preds[["rating"]], self._y_test[["rating"]]
        )
        sq_error[slice] = self.product_data.loc[self._x_test["movie_id"]][slice].values
        return sq_error.explode(slice).groupby(slice)["rating"].agg("mean")

    @rec_test(test_type="stats")
    def basic_stats(self):
        return self._y_preds.describe()

    @rec_test(test_type="rmse")
    def rmse(self):
        return (
            self.squared_error(self._y_preds["rating"], self._y_test["rating"]).mean()
            ** 0.5
        )

    @rec_test(test_type="rmse_by_genre")
    def rmse_by_genre(self):
        return self.rmse_by_slice("genres")


class CoveoCartRecList(RecList):
    @rec_test(test_type="stats")
    def basic_stats(self):
        """
        Basic statistics on training, test and prediction data
        """
        from reclist.metrics.standard_metrics import statistics

        return statistics(
            self._x_train, self._y_train, self._x_test, self._y_test, self._y_preds
        )

    @rec_test(test_type="price_homogeneity")
    def price_test(self):
        """
        Measures the absolute log ratio of ground truth and prediction price
        """
        from reclist.metrics.price_homogeneity import price_homogeneity_test

        return price_homogeneity_test(
            y_test=self.sku_only(self._y_test),
            y_preds=self.sku_only(self._y_preds),
            product_data=self.product_data,
            price_sel_fn=lambda x: float(x["price_bucket"])
            if x["price_bucket"]
            else None,
        )

    @rec_test(test_type="Coverage@10")
    def coverage_at_k(self):
        """
        Coverage is the proportion of all possible products which the RS
        recommends based on a set of sessions
        """
        from reclist.metrics.standard_metrics import coverage_at_k

        return coverage_at_k(self.sku_only(self._y_preds), self.product_data, k=10)

    @rec_test(test_type="HR@10")
    def hit_rate_at_k(self):
        """
        Compute the rate in which the top-k predictions contain the item to be predicted
        """
        from reclist.metrics.standard_metrics import hit_rate_at_k

        return hit_rate_at_k(
            self.sku_only(self._y_preds), self.sku_only(self._y_test), k=10
        )

    @rec_test(test_type="hits_distribution")
    def hits_distribution(self):
        """
        Compute the distribution of hit-rate across product frequency in training data
        """
        from reclist.metrics.hits import hits_distribution

        return hits_distribution(
            self.sku_only(self._x_train),
            self.sku_only(self._x_test),
            self.sku_only(self._y_test),
            self.sku_only(self._y_preds),
            k=10,
            debug=True,
        )

    @rec_test(test_type="distance_to_query")
    def dist_to_query(self):
        """
        Compute the distribution of distance from query to label and query to prediction
        """
        from reclist.metrics.distance_metrics import distance_to_query

        return distance_to_query(
            self.rec_model,
            self.sku_only(self._x_test),
            self.sku_only(self._y_test),
            self.sku_only(self._y_preds),
            k=10,
            bins=25,
            debug=True,
        )

    def sku_only(self, l: List[List]):
        return [[e["product_sku"] for e in s] for s in l]


class SpotifySessionRecList(RecList):
    @rec_test(test_type="basic_stats")
    def basic_stats(self):
        """
        Basic statistics on training, test and prediction data for Next Event Prediction
        """
        from reclist.metrics.standard_metrics import statistics

        return statistics(
            self._x_train, self._y_train, self._x_test, self._y_test, self._y_preds
        )

    @rec_test(test_type="HR@10")
    def hit_rate_at_k(self):
        """
        Compute the rate at which the top-k predictions contain the item to be predicted
        """
        from reclist.metrics.standard_metrics import hit_rate_at_k

        return hit_rate_at_k(
            self.uri_only(self._y_preds), self.uri_only(self._y_test), k=10
        )

    @rec_test(test_type="perturbation_test")
    def perturbation_at_k(self):
        """
        Compute average consistency in model predictions when inputs are perturbed
        """
        from collections import defaultdict
        from functools import partial

        from reclist.metrics.perturbation import session_perturbation_test

        # Step 1: Generate a map from artist uri to track uri
        substitute_mapping = defaultdict(list)
        for track_uri, row in self.product_data.items():
            substitute_mapping[row["artist_uri"]].append(track_uri)

        # Step 2: define a custom perturbation function
        def perturb(session, sub_map):
            last_item = session[-1]
            last_item_artist = self.product_data[last_item["track_uri"]]["artist_uri"]
            substitutes = set(sub_map.get(last_item_artist, [])) - {
                last_item["track_uri"]
            }
            if substitutes:
                similar_item = random.sample(substitutes, k=1)
                new_session = session[:-1] + [{"track_uri": similar_item[0]}]
                return new_session
            return []

        # Step 3: call test
        return session_perturbation_test(
            self.rec_model,
            self._x_test,
            self._y_preds,
            partial(perturb, sub_map=substitute_mapping),
            self.uri_only,
            k=10,
        )

    @rec_test(test_type="shuffle_session")
    def perturbation_shuffle_at_k(self):
        """
        Compute average consistency in model predictions when inputs are re-ordered
        """
        from reclist.metrics.perturbation import session_perturbation_test

        # Step 1: define a custom perturbation function
        def perturb(session):
            return random.sample(session, len(session))

        # Step 2: call test
        return session_perturbation_test(
            self.rec_model, self._x_test, self._y_preds, perturb, self.uri_only, k=10
        )

    @rec_test(test_type="hits_distribution_by_slice")
    def hits_distribution_by_slice(self):
        """
        Compute the distribution of hit-rate across various slices of data
        """
        from reclist.metrics.hits import hits_distribution_by_slice

        len_map = collections.defaultdict(list)
        for idx, playlist in enumerate(self._x_test):
            len_map[len(playlist)].append(idx)
        slices = collections.defaultdict(list)
        bins = [(x * 5, (x + 1) * 5) for x in range(max(len_map) // 5 + 1)]
        for bin_min, bin_max in bins:
            for i in range(bin_min + 1, bin_max + 1, 1):
                slices[f"({bin_min}, {bin_max}]"].extend(len_map[i])
                del len_map[i]
        assert len(len_map) == 0

        return hits_distribution_by_slice(
            slices,
            self.uri_only(self._y_test),
            self.uri_only(self._y_preds),
            debug=True,
        )

    @rec_test(test_type="Coverage@10")
    def coverage_at_k(self):
        """
        Coverage is the proportion of all possible products which the RS
        recommends based on a set of sessions
        """
        from reclist.metrics.standard_metrics import coverage_at_k

        return coverage_at_k(
            self.uri_only(self._y_preds),
            self.product_data,
            # this contains all the track URIs from train and test sets
            k=10,
        )

    @rec_test(test_type="Popularity@10")
    def popularity_bias_at_k(self):
        """
        Compute average frequency of occurrence across recommended items in training data
        """
        from reclist.metrics.standard_metrics import popularity_bias_at_k

        return popularity_bias_at_k(
            self.uri_only(self._y_preds), self.uri_only(self._x_train), k=10
        )

    @rec_test(test_type="MRR@10")
    def mrr_at_k(self):
        """
        MRR calculates the mean reciprocal of the rank at which the first
        relevant item was retrieved
        """
        from reclist.metrics.standard_metrics import mrr_at_k

        return mrr_at_k(self.uri_only(self._y_preds), self.uri_only(self._y_test))

    def uri_only(self, playlists: List[dict]):
        return [[track["track_uri"] for track in playlist] for playlist in playlists]


class MovieLensSimilarItemRecList(RecList):
    @rec_test(test_type="stats")
    def basic_stats(self):
        """
        Basic statistics on training, test and prediction data
        """
        from reclist.metrics.standard_metrics import statistics

        return statistics(
            self._x_train, self._y_train, self._x_test, self._y_test, self._y_preds
        )

    @rec_test(test_type="HR@10")
    def hit_rate_at_k(self):
        """
        Compute the rate at which the top-k predictions contain the movie to be predicted
        """
        from reclist.metrics.standard_metrics import hit_rate_at_k

        return hit_rate_at_k(
            self.movie_only(self._y_preds), self.movie_only(self._y_test), k=10
        )

    @rec_test(test_type="Coverage@10")
    def coverage_at_k(self):
        """
        Coverage is the proportion of all possible movies which the RS
        recommends based on a set of movies and their respective ratings
        """
        from reclist.metrics.standard_metrics import coverage_at_k

        return coverage_at_k(self.movie_only(self._y_preds), self.product_data, k=10)

    @rec_test(test_type="hits_distribution")
    def hits_distribution(self):
        """
        Compute the distribution of hit-rate across movie frequency in training data
        """
        from reclist.metrics.hits import hits_distribution

        return hits_distribution(
            self.movie_only(self._x_train),
            self.movie_only(self._x_test),
            self.movie_only(self._y_test),
            self.movie_only(self._y_preds),
            k=10,
            debug=True,
        )

    @rec_test(test_type="hits_distribution_by_rating")
    def hits_distribution_by_rating(self):
        """
        Compute the distribution of hit-rate across movie ratings in testing data
        """
        from reclist.metrics.hits import hits_distribution_by_rating

        return hits_distribution_by_rating(self._y_test, self._y_preds, debug=True)

    def movie_only(self, movies):
        return [[x["movieId"] for x in y] for y in movies]


class SyntheticRatingClassifierRecList(RecList):
    def __init__(self, model, dataset):
        super().__init__(model, dataset)
        self._y_preds.columns = self._y_test.columns
        self.name = "SyntheticRating" + "_" + "Classifier"

    @rec_test(test_type="min_stats")
    def statistics(self):
        statistics = {}
        statistics["x_test"] = self._x_test.describe(include="all").to_json()
        statistics["x_train"] = self._x_train.describe(include="all").to_json()
        statistics["y_train"] = self._y_train.describe(include="all").to_json()
        statistics["y_test"] = self._y_test.describe(include="all").to_json()
        statistics["y_preds"] = self._y_preds.describe(include="all").to_json()
        return statistics

    @rec_test(test_type="rmse")
    def rmse(self):
        from sklearn.metrics import mean_squared_error

        return mean_squared_error(
            self._y_test, self._y_preds, squared=False, multioutput="uniform_average"
        )

    @rec_test(test_type="rmse_by_user")
    def rmse_by_groups(self, group_by_columns=["user_id"]):
        from sklearn.metrics import mean_squared_error

        temp = pd.DataFrame()
        temp["user_id"] = self._x_test["user_id"]
        temp["y_test"] = self._y_test["interactions"]
        temp["y_pred"] = self._y_preds["interactions"]
        rmse_user = temp.groupby(group_by_columns).apply(
            lambda x: mean_squared_error(
                x["y_test"], x["y_pred"], squared=False, multioutput="uniform_average"
            )
        )
        del temp
        return rmse_user.to_json()

    @rec_test(test_type="classification_report")
    def classification_report(self, k: int = 3):
        import sklearn

        return sklearn.metrics.classification_report(self._y_test, self._y_preds)

    @rec_test(test_type="rmse@3")
    def rmse_at_k(self, k: int = 3, filter_by="interactions", filter_func=None):
        from sklearn.metrics import mean_squared_error

        filter_ = self._y_test[filter_by] >= k
        if filter_.sum() == 0:
            return 0
        return mean_squared_error(
            self._y_test[filter_],
            self._y_preds[filter_],
            squared=False,
            multioutput="uniform_average",
        )

    @rec_test(test_type="rmse@k-all")
    def rmse_at_k_all(self, k: int = 5):
        rmse_at_k = {}
        for i in range(1, k):
            rmse_at_k[f"rmse@{i}"] = self.rmse_at_k(i)
        return rmse_at_k

    @plot_test_model(plot_name="scatter_true_pred")
    def plot_true_pred(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.scatter(self._y_test.values, self._y_preds.values)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title("True vs Predicted")

        return fig

    @plot_test_model(plot_name="rmse_by_user")
    def plot_rmse_by_user(self):
        import json

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for test in self.test_results:
            if test["test_name"] == "rmse_by_user":
                test_result = test["test_result"]
                break
        test_result = json.loads(test_result)
        ax.set_xlabel("User")
        ax.set_ylabel("RMSE")
        ax.set_title("RMSE by User")
        ax.bar(test_result.keys(), test_result.values())
        return fig

    @plot_test_model(plot_name="rmse@k-all")
    def plot_rmse_at_k_all(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        for test in self.test_results:
            if test["test_name"] == "rmse@k-all":
                test_result = test["test_result"]
                break
        ax.set_xlabel("k")
        ax.set_ylabel("RMSE")
        ax.set_title("RMSE at k")
        ax.plot(test_result.keys(), test_result.values())
        return fig

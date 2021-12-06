import numpy as np
from reclist.abstractions import RecModel
from reclist.utils.train_w2v import train_embeddings


class CoveoP2VRecModel(RecModel):
    """
    Implement of the prod2vec model through the standard RecModel interface.

    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    model_name = "prod2vec"

    def train(self, products, iterations=15):
        """
        Takes a list of products in the COVEO dataset (with coveo dataset format) and trains a model

        :param products: list of json
        :return:
        """
        x_train_skus = [[e['product_sku'] for e in s] for s in products]
        self._model = train_embeddings(x_train_skus, iterations=iterations)

    def predict(self, prediction_input: list, *args, **kwargs):
        """
        Implement the abstract method, accepting a list of lists, each list being
        the content of a cart: the predictions returned by the model are the top K
        items suggested to complete the cart.

        :param prediction_input:
        :return:

        """
        predictions = []
        for _x in prediction_input:
            # we assume here that every X is a list of one-element, the product already in the cart
            # i.e. our prediction_input list is [[sku_1], [sku_3], ...]
            key_item = _x[0]['product_sku']
            nn_products = self._model.most_similar(key_item, topn=10) if key_item in self._model else None
            if nn_products:
                predictions.append([{'product_sku': _[0]} for _ in nn_products])
            else:
                predictions.append([])

        return predictions

    def get_vector(self, product_sku):
        try:
            return list(self._model.get_vector(product_sku))
        except Exception as e:
            return []


class SpotifyP2VRecModel(RecModel):
    """
    Implement of the prod2vec model through the standard RecModel interface.

    Since init is ok, we just need to overwrite the prediction methods to get predictions
    out of it.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    model_name = "prod2vec"

    def train(self, playlists, iterations=15):
        x_train_uris = [[track['track_uri'] for track in playlist] for playlist in playlists]
        self._model = train_embeddings(x_train_uris, iterations=iterations)

    def predict(self, prediction_input: list, *args, **kwargs):
        """
        Implement the abstract method.
        NEP following industry best practices mentioned in https://arxiv.org/abs/2007.14906:
        given a trained prod2vec, take all the seeded (or before-last) tracks to construct a
        playlist vector by average pooling, and use kNN to predict the next items (or the last
        item).
        """
        predictions = []
        for _x in prediction_input:
            embeddings = [self._model[track['track_uri']] for track in _x if track['track_uri'] in self._model]
            if embeddings:
                avg_playlist_vector = np.average(embeddings, axis=0)
                nn_tracks = self._model.similar_by_vector(avg_playlist_vector, topn=100)
                predictions.append([{'track_uri':_[0]} for _ in nn_tracks])
            else:
                predictions.append([])

        return predictions

    def get_vector(self, track_uri):
        try:
            return list(self._model.get_vector(track_uri))
        except Exception as e:
            return []

class MovieLensP2VRecModel(RecModel):
    """
    Prod2Vec implementation for MovieLens 25M dataset
    """
    model_name = "prod2vec"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, movies, iterations=15):
        # Get the movie ID and rating for each movie for each unique user
        x_train = [[(x["movieId"], x["rating"]) for x in y] for y in movies]
        self._model = train_embeddings(x_train, iterations=iterations)

    def predict(self, prediction_input, *args, **kwargs):
        """
        Predicts the top 10 similar items recommended for each user according
        to the movies that they've watched and the ratings that they've given

        :param prediction_input: a list of lists containing a dictionary for
                                 each movie watched by that user
        :return:
        """
        all_predictions = []
        for movies in prediction_input:
            predictions = []
            emb_vecs = []
            for movie in movies:
                emb_vec = self.get_vector(movie)
                if emb_vec:
                    emb_vecs.append(emb_vec)
            if emb_vecs:
                # Calculate the average of all the latent vectors representing
                # the movies watched by the user as is done in https://arxiv.org/abs/2007.14906
                avg_emb_vec = np.mean(emb_vecs, axis=0)
                nn_products = self.model.similar_by_vector(avg_emb_vec, topn=10)
                for elem in nn_products:
                    predictions.append({"movieId": elem[0][0], "rating": elem[0][1]})
            all_predictions.append(predictions)
        return all_predictions

    def get_vector(self, x):
        """
        Returns the latent vector that corresponds to the movie ID and the rating

        :param x:
        :return:
        """
        movie = (x["movieId"], x["rating"])
        try:
            return list(self.model.get_vector(movie))
        except Exception as e:
            return []

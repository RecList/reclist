from reclist.abstractions import RecModel
from reclist.utils.train_w2v import train_embeddings


class CoveoP2VRecModel(RecModel):
    """
    Implement of the prod2vec model through the standard RecModel interface.

    >>> model = CoveoP2VRecModel()
    >>> model.train(coveo_dataset.x_train)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    model_name = "prod2vec"

    def train(self, products):
        """
        Takes a list of products in the COVEO dataset (with coveo dataset format) and trains a model

        :param products: list of json
        :return:
        """
        x_train_skus = [[e['product_sku'] for e in s] for s in products]
        self._model = train_embeddings(x_train_skus)

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
                predictions.append([{'product_sku':_[0]} for _ in nn_products])
            else:
                predictions.append([])

        return predictions

    def get_vector(self, product_sku):
        try:
            return list(self._model.get_vector(product_sku))
        except Exception as e:
            return []

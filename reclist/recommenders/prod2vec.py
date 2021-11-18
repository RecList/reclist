from reclist.abstractions import RecModel

class P2VRecModel(RecModel):
    """
    Implement of the prod2vec model through the standard RecModel interface.

    Since init is ok, we just need to overwrite the prediction methods to get predictions
    out of it.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    model_name = "prod2vec"
    def predict(self, prediction_input: list, *args, **kwargs):
        """
        Implement the abstract method, accepting a list of lists, each list being
        the content of a cart: the predictions returned by the model are the top K
        items suggested to complete the cart.
        """
        predictions = []
        for _x in prediction_input:
            # we assume here that every X is a list of one-element, the product already in the cart
            # i.e. our prediction_input list is [[sku_1], [sku_3], ...]
            key_item = _x[0]['product_sku']
            nn_products = self._model.most_similar(key_item, topn=3) if key_item in self._model else None
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

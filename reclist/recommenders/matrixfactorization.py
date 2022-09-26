import pandas as pd
from matrix_factorization import KernelMF

from reclist.abstractions import RecModel


class MovieLensKernelMFModel(RecModel):
    """
    KernelMF implementation for MovieLens 25M dataset
    """

    model_name = "KernelMF"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(
        self,
        user_movie_pairs: pd.DataFrame,
        ratings: pd.DataFrame,
        user_column_name: str = "user_id",
        movie_column_name: str = "movie_id",
        rating_column_name: str = "rating",
    ):
        # init KernelMF object
        self._model = matrix_fact = KernelMF(
            n_epochs=20, n_factors=100, verbose=1, lr=0.001, reg=0.005
        )
        # re-order columns
        x_train = user_movie_pairs[[user_column_name, movie_column_name]]
        # rename columns as per KernelMF requirements
        x_train.columns = ["user_id", "item_id"]
        # get ratings as pd.Series
        y_train = ratings[rating_column_name]
        matrix_fact.fit(x_train, y_train)

    def predict(
        self,
        user_movie_pairs: pd.DataFrame,
        user_column_name: str = "user_id",
        movie_column_name: str = "movie_id",
        *args,
        **kwargs
    ) -> pd.DataFrame:
        """
        Predicts the user rating for a given movie

        """
        x_test = user_movie_pairs[[user_column_name, movie_column_name]]
        x_test.columns = ["user_id", "item_id"]

        return pd.DataFrame(
            self.model.predict(x_test), columns=["rating"], index=x_test.index
        )

Core Components
===============

To understand how to use RecList you need to know the core components of the package.
After reading this page you should be comfortable enough to play and modify the tutorial and also create your
reclists for evaluation.

Two Paths
~~~~~~~~~


Say you have a new recommender and you want to validate on behavioural tests. We have different datasets you can play
with. If you want to follow this path, you just need to get the training data from the datastets, train the model and then
run the reclist to evaluate your model.

Another possible use case is that you might want to create a RecList for a new dataset you have.

RecTest
~~~~~~~

The RecTest is probably the most fundamental abstraction in RecList. The RecTest is essentially a decorator used to
evaluate the various datasets.


.. code-block:: python

    class MyRecListRecList(RecList):

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


RecDataset
~~~~~~~~~~

The dataset is a simple abstraction you might want to implement if you are playing with other datasets.
You will see that we just need to instantiate the main parameters.


.. code-block:: python

    class RecDataset(ABC):
    """
    Implements an abstract class for the dataset
    """
    def __init__(self, force_download=False):
        """
        :param force_download: allows to force the download of the dataset in case it is needed.
        :type: force_download: bool, optional
        """
        self._x_train = None
        self._y_train = None
        self._x_test = None
        self._y_test = None
        self._catalog = None
        self.force_download = force_download
        self.load()

    @abstractmethod
    def load(self):
        """
        Abstract method that should implement dataset loading
        @return:
        """
        return


RecModel
~~~~~~~~



.. code-block:: python

    class MyRecModel(RecModel):
        """
        My Recommender Model
        """
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        model_name = "mymodel"

        def train(self, products):

            self._model = my_training_function(products)

        def predict(self, prediction_input: list, *args, **kwargs):

            predictions = self.model.predict(prediction_input)

            return predictions

        def get_vector(self, product_sku):
            try:
                return list(self._model.get_vector(product_sku))
            except Exception as e:
                return []

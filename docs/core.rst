Core Components
===============


To understand how to use RecList you need to know the core components of the package.
After reading this page you should be comfortable enough to play and modify the tutorial and also create your
reclists for evaluation.

Two Paths
~~~~~~~~~


Say you have a new recommender and you want to validate on behavioural tests. We have different datasets you can play
with.

Another possible use case is that you might want to create a RecList for a new dataset you have.

RecTest
-------

The RecTest is probably the most fundamental abstraction in RecList.


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
----------

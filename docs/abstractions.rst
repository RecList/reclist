=======
RecTest
=======

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

============
Abstractions
============

.. automodule:: reclist.abstractions
    :members:
    :undoc-members:
    :show-inheritance:





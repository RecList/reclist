=======
RecList
=======


.. image:: https://img.shields.io/pypi/v/reclist.svg
        :target: https://pypi.python.org/pypi/reclist

.. image:: https://readthedocs.org/projects/reclist/badge/?version=latest
        :target: https://reclist.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status

.. image:: https://github.com/jacopotagliabue/reclist/workflows/Python%20package/badge.svg
        :target: https://github.com/jacopotagliabue/reclist/actions

.. image:: https://img.shields.io/github/contributors/jacopotagliabue/reclist
        :target: https://github.com/jacopotagliabue/reclist/graphs/contributors/
        :alt: Contributors

.. image:: https://img.shields.io/badge/License-MIT-blue.svg
        :target: https://lbesson.mit-license.org/
        :alt: License

.. image:: https://pepy.tech/badge/reclist
        :target: https://pepy.tech/project/reclist
        :alt: Downloads

.. image:: https://img.shields.io/badge/youtube-video-red
        :target: https://www.youtube.com/watch?v=cAlJYxFYA04
        :alt: YouTube

.. image:: images/reclist.png
        :width: 50%
        :alt: Rocket Emoji


Overview
--------

*RecList* is an open source library providing behavioral, "black-box" testing for recommender systems. Inspired by the pioneering work of
`Ribeiro et al. 2020 <https://aclanthology.org/2020.acl-main.442.pdf>`__ in NLP, we introduce a general plug-and-play procedure to scale up behavioral testing, with an easy-to-extend interface for custom use cases.

While quantitative metrics over held-out data points are important, a lot more tests are needed for recommenders
to properly function in the wild and not erode our confidence in them: for example, a model may boast an accuracy improvement over the entire dataset, but actually be significantly worse than another on rare items or new users; or again, a model that correctly recommends HDMI cables as add-on for shoppers buying a TV, may also wrongly  recommend TVs to shoppers just buying a cable.

*RecList* goal is to operationalize these important intuitions into a practical package for testing research and production models in a more nuanced way, without
requiring unnecessary custom code and ad hoc procedures.

If you are not familiar with the library, we suggest first taking our small tour to get acquainted with the main abstractions through ready-made models and tests.

Quick Links
~~~~~~~~~~~

* The original `paper <https://dl.acm.org/doi/abs/10.1145/3487553.3524215>`__ (`arxiv <https://arxiv.org/abs/2111.09963>`__) and initial `release post <https://towardsdatascience.com/ndcg-is-not-all-you-need-24eb6d2f1227>`__.
* `EvalRS22@CIKM <https://github.com/RecList/evalRS-CIKM-2022>`__ and `EvalRS23@KDD <https://reclist.io/kdd2023-cup/>`__ , for music recommendations with RecList.
* A `colab notebook <https://colab.research.google.com/drive/1Wn5mm0csEkyWqmBBDxNBkfGR6CNfWeH-?usp=sharing>`__, for a quick interactive tour.
* Our `website <https://reclist.io/>`__ for past talks and presentations.


Status
~~~~~~~~~~~

RecList is free software released under the MIT license, and it has been adopted by popular `open-source <https://github.com/RecList/evalRS-CIKM-2022>`__  `data challenges <https://reclist.io/kdd2023-cup/>`__.

The current version, after a major API re-factoring, is a *beta*.

Summary
~~~~~~~

This doc is structured as follows:

* `Quick Start`_
* `A Guided Tour`_
* `Capabilities`_
* `License and Citation`_
* `Acknowledgments`_

Quick Start
-----------

If you want to see *RecList* in action, clone the repository, create and activate a virtual env, and install the required packages from pip (you can install from root of course). If you prefer to experiment in an interactive, no-installation-required fashion, try out our `colab notebook <https://colab.research.google.com/drive/1Wn5mm0csEkyWqmBBDxNBkfGR6CNfWeH-?usp=sharing>`__.

.. code-block:: bash

    git clone https://github.com/jacopotagliabue/reclist
    cd reclist
    python3 -m venv venv
    source venv/bin/activate
    pip install reclist
    python examples/evalrs_2023.py

Once you've run successfully the sample script, take the guided tour below to learn more about the abstractions and the out-of-the-box capabilities of *RecList*.

A Guided Tour
-------------

An instance of `RecList <https://github.com/jacopotagliabue/reclist/blob/main/reclist/reclist.py>`__ represents a suite of tests for recommender systems.

As the sample `examples/evalrs_2023.py` script shows, we leave users quite a large range of options when building a RecList instance: we provide out of the box standard metrics
in case your dataset is DataFrame-shaped (or you can / wish turn it into such a shape), but don't force you any pattern if you just want to use RecList
for the scaffolding it provides.

For example, the following code only assumes you have a dataset with golden labels, predictions, and metadata (e.g. item features) in the form of a DataFrame:

.. code-block:: python

    cdf = DFSessionRecList(
        dataset=df_events,
        model_name="myDataFrameRandomModel",
        predictions=df_predictions,
        y_test=df_dataset,
        logger=LOGGER.NEPTUNE,
        metadata_store= METADATA_STORE.LOCAL,
        similarity_model=my_sim_model,
        bucket=os.environ["S3_BUCKET"],
        NEPTUNE_KEY=os.environ["NEPTUNE_KEY"],
        NEPTUNE_PROJECT_NAME=os.environ["NEPTUNE_PROJECT_NAME"],
    )

    # run reclist
    cdf(verbose=True)

Our library pre-packages standard recSys KPIs and important behavioral tests, but it is built with extensibility in mind: you can re-use tests in new suites, or you can write new domain-specific suites and tests.

Any suite must inherit the *RecList* interface, and then declare with Pytonic decorators its tests. In this case, the test re-uses a standard function:

.. code-block:: python

    class MyRecList(RecList):

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


Any model can be tested, as RecList does not make any assumption on the model's internal structure, but only the availability of `predictions`
and `ground truth`. Once again, while our example leverages a DataFrame-shaped dataset for these entities, you are free to build your own
RecList instance with any shape you prefer, provided you implement the metrics accordingly.

*RecList* recognizes that outside of academic benchmarks, some mistakes are worse than others, and not all inputs are created equal: when possible, it tries
to operationalize through scalable code behavioral insights for debugging and error analysis; it also provides extensible abstractions when domain knowledge and custom logic are needed.

Once you run a suite of tests, results are dumped automatically and versioned in a folder (local or on S3), structured as follows
(name of the suite, name of the model, run timestamp):

.. code-block::

    .reclist/
      myList/
        myModel/
          1637357392/
          1637357404/

If you start using *RecList* as part of your standard testings - either for research or production purposes - you can use the JSON report
for machine-to-machine communication with downstream system (e.g. you may want to automatically fail the model pipeline if certain behavioral tests are not passed).

Capabilities
------------

*RecList* provides a dataset and model agnostic framework to scale up behavioral tests. We provide some suggested abstractions
based on DataFrames to make existing tests and metrics fully re-usable, but we don't force any pattern on the user. As out of the box functionality, the package provides:

* tests and metrics to be used on your own datasets and models;

* automated storage of results, with versioning, both in a local folder or on S3 in the cloud;

* pre-built connectors with popular experiment trackers (Neptune, Comet), and an extensible interface to add your own;

* reference implementations based on popular data challenges that used RecList.


Acknowledgments
---------------

The original authors of RecList are:

* Patrick John Chia - `LinkedIn <https://www.linkedin.com/in/patrick-john-chia-b0a34019b/>`__, `GitHub <https://github.com/patrickjohncyh>`__
* Jacopo Tagliabue - `LinkedIn <https://www.linkedin.com/in/jacopotagliabue/>`__, `GitHub <https://github.com/jacopotagliabue>`__
* Federico Bianchi - `LinkedIn <https://www.linkedin.com/in/federico-bianchi-3b7998121/>`__, `GitHub <https://github.com/vinid>`__
* Chloe He - `LinkedIn <https://www.linkedin.com/in/chloe-he//>`__, `GitHub <https://github.com/chloeh13q>`__
* Brian Ko - `LinkedIn <https://www.linkedin.com/in/briankosw/>`__, `GitHub <https://github.com/briankosw>`__

RecList beta has been developed with the help of:

* Unnati Patel - `LinkedIn <https://www.linkedin.com/in/unnati-p-16626610a/>`__

If you have questions or feedback, please reach out to: :code:`jacopo dot tagliabue at nyu dot edu`.


Supporters
-----------------------
RecList is a community project made possible by the generous support of awesome folks. Between June and December 2022, the development of our beta has been supported by `Comet <https://www.comet.com/>`__, `Neptune <https://neptune.ai/homepage>`__ , `Gantry <https://gantry.io/>`__.


License and Citation
--------------------

All the code is released under an open MIT license. If you found *RecList* useful, please cite our WWW paper:

.. code-block:: bash

    @inproceedings{10.1145/3487553.3524215,
        author = {Chia, Patrick John and Tagliabue, Jacopo and Bianchi, Federico and He, Chloe and Ko, Brian},
        title = {Beyond NDCG: Behavioral Testing of Recommender Systems with RecList},
        year = {2022},
        isbn = {9781450391306},
        publisher = {Association for Computing Machinery},
        address = {New York, NY, USA},
        url = {https://doi.org/10.1145/3487553.3524215},
        doi = {10.1145/3487553.3524215},
        pages = {99–104},
        numpages = {6},
        keywords = {recommender systems, open source, behavioral testing},
        location = {Virtual Event, Lyon, France},
        series = {WWW '22 Companion}
    }

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

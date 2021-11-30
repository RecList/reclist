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

RecList


* Free software: MIT license
* Documentation: https://reclist.readthedocs.io.

Overview
--------

*RecList* is an open source library providing behavioral, "black-box" testing for recommender systems. Inspired by the pioneering work of
`Ribeiro et al. 2020 <https://aclanthology.org/2020.acl-main.442.pdf>`__ in NLP, we introduce a general plug-and-play procedure to scale up behavioral testing,
with an easy-to-extend interface for custom use cases.

*RecList* ships with some popular datasets and ready-made behavioral tests: check the `paper <https://arxiv.org/abs/2111.09963>`__
for more details on the relevant literature and the philosophical motivations behind the project.

If you are not familiar with the library, we suggest first taking our small tour to get acquainted with the main
abstractions through ready-made models and public datasets.

Quick Links
~~~~~~~~~~~

* Our `paper <https://arxiv.org/abs/2111.09963>`__, with in-depth analysis, detailed use cases and scholarly references.
* A `colab notebook <https://colab.research.google.com/drive/1Wn5mm0csEkyWqmBBDxNBkfGR6CNfWeH-?usp=sharing>`__ (WIP), showing how to train a cart recommender model from scratch and use the library to test it.
* Our blog post (forthcoming), with examples and practical tips. 

Project updates
~~~~~~~~~~~~~~~

*Nov. 2021*: the library is currently in alpha (i.e. enough working code to finish the paper and tinker with it). We welcome early feedback, but please be advised that the package may change substantially in the upcoming months ("If you're not embarrassed by the first version, you've launched too late").

As the project is in active development, come back often for updates.

Summary
~~~~~~~

This doc is structured as follows:

* `Quick Start`_
* `A Guided Tour`_
* `Capabilities`_
* `Roadmap`_
* `Acknowledgments`_
* `License and Citation`_

Quick Start
-----------

If you want to see *RecList* in action, clone the repository, create and activate a virtual env, and install
the required packages from root. If you prefer to experiment in an interactive, no-installation-required fashion,
try out our `colab notebook <https://colab.research.google.com/drive/1Wn5mm0csEkyWqmBBDxNBkfGR6CNfWeH-?usp=sharing>`__.

Sample scripts are divided by use-cases: similar items, complementary items or
session-based recommendations. When executing one, a suitable public dataset will be downloaded,
and a baseline ML model trained: finally, the script will run a pre-made suite of behavioral tests
to show typical results.

.. code-block:: bash

    git clone https://github.com/jacopotagliabue/reclist
    cd reclist
    python3 -m venv venv
    source venv/bin/activate
    pip install -e .
    python examples/coveo_complementary_rec.py

Running *your* model on one of the supported dataset, leveraging the pre-made tests, is as easy as implementing
a simple interface, *RecModel*.

Once you've run successfully the sample script, take the guided tour below to learn more about
the abstractions and the out-of-the-box capabilities of *RecList*.

A Guided Tour
-------------

An instance of `RecList <https://github.com/jacopotagliabue/reclist/blob/main/reclist/reclist.py>`__ represents a suite of tests for recommender systems: given
a dataset (more appropriately, an instance of `RecDataset <https://github.com/jacopotagliabue/reclist/blob/main/reclist/abstractions.py>`__)
and a model (an instance of `RecModel <https://github.com/jacopotagliabue/reclist/blob/main/reclist/abstractions.py>`__), it will
run the specified tests on the target dataset, using the supplied model. 

For example, the following code instantiates a pre-made suite of tests that contains sensible defaults 
for a `cart recommendation use case <https://github.com/jacopotagliabue/reclist/blob/main/reclist/reclist.py>`__:

.. code-block:: python
   
    rec_list = CoveoCartRecList(
        model=model,
        dataset=coveo_dataset
    )
    # invoke rec_list to run tests
    rec_list(verbose=True)

Our library pre-packages standard recSys KPIs and important behavioral tests, divided by use cases, but it is built with 
extensibility in mind: you can re-use tests in new suites, or you can write new domain-specific suites and tests. 

Any suite must inherit the *RecList* interface, and then declare with Pytonic decorators its tests: 
in this case, the test re-uses a standard function:

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


Any model can be tested, as long as its predictions are wrapped in a *RecModel*. This allows for pure "black-box" testings, 
a SaaS provider can be tested just by wrapping the proper API call in the method:

.. code-block:: python
   
    class MyCartModel(RecModel):
    
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def predict(self, prediction_input: list, *args, **kwargs):
            """
            Implement the abstract method, accepting a list of lists, each list being
            the content of a cart: the predictions returned by the model are the top K
            items suggested to complete the cart.
            """

            return

While many standard KPIs are available in the package, the philosophy behind *RecList* is that metrics like Hit Rate provide only a partial picture
of the expected behavior of recommenders in the wild: two models with very similar accuracy can have very different behavior on, say, the long-tail, or
model A can be better than model B overall, but at the expense of providing disastrous performance on a set of inputs that are particularly important in production. 

*RecList* recognizes that outside of academic benchmarks, some mistakes are worse than others, and not all inputs are created equal: when possible, it tries
to operationalize through scalable code behavioral insights for debugging and error analysis; it also
provides extensible abstractions when domain knowledge and custom logic are needed.

Once you run a suite of tests, results are dumped automatically and versioned in a local folder, structured as follows
(name of the suite, name of the model, run timestamp):

.. code-block:: 

    .reclist/
      myList/
        myModel/
          1637357392/
          1637357404/

We provide a simple (and *very* WIP) UI to easily compare runs and models. After you run two times one of the example scripts,
you can do:

.. code-block:: bash

    cd app
    python app.py

to start a local web app that lets you explore test results:    

.. image:: https://github.com/jacopotagliabue/reclist/blob/main/images/explorer.png
   :height: 200

If you select more than model, the app will automatically build comparison tables:

.. image:: https://github.com/jacopotagliabue/reclist/blob/main/images/comparison.png
   :height: 200

If you start using *RecList* as part of your standard testings - either for research or production purposes - you can use the JSON report
for machine-to-machine communication with downstream system (e.g. you may want to automatically fail the model pipeline if certain behavioral tests are not passed).

Capabilities
------------

*RecList* provides a dataset and model agnostic framework to scale up behavioral tests. As long as the proper abstractions
are implemented, all the out-of-the-box components can be re-used. For example:

* you can use a public dataset provided by *RecList* to train your new cart recommender model, and then use the *RecTests* we provide for that use case;

* you can use some baseline model on your custom dataset, to establish a baseline for your project;

* you can use a custom model, on a private dataset and define from scratch a new suite of tests, mixing existing methods and domain-specific tests.

We list below what we currently support out-of-the-box, with particular focus on datasets and tests, as the models we provide
are convenient baselines, but they are not meant to be SOTA research models.

Datasets
~~~~~~~~

RecList features convenient wrappers around popular datasets, to help test models over known benchmarks
in a standardized way.

* `Coveo Data Challenge <https://github.com/coveooss/SIGIR-ecom-data-challenge>`__
* `The Million Playlist Dataset <https://engineering.atspotify.com/2018/05/30/introducing-the-million-playlist-dataset-and-recsys-challenge-2018/>`__ (*coming soon*)
* `MovieLens <https://grouplens.org/datasets/movielens/>`__ (*coming soon*)

Behavioral Tests
~~~~~~~~~~~~~~~~

*Coming soon!*

Roadmap
-------

To do:

* the app is just a stub: improve the report "contract" and extend the app capabilities, possibly including it in the library itself;

* continue adding default *RecTests* by use cases, and test them on public datasets;

* improving our test suites and refactor some abstractions;

* adding Colab tutorials, extensive documentation and a blog-like write-up to explain the basic usage.

We maintain a small Trello board on the project which we plan on sharing with the community: *more details coming soon*!

Contributing
~~~~~~~~~~~~

We will update this repo with some guidelines for contributions as soon as the codebase becomes more stable.
Check back often for updates!

Acknowledgments
---------------

The main contributors are:

* Patrick John Chia - `LinkedIn <https://www.linkedin.com/in/patrick-john-chia-b0a34019b/>`__, `GitHub <https://github.com/patrickjohncyh>`__
* Jacopo Tagliabue - `LinkedIn <https://www.linkedin.com/in/jacopotagliabue/>`__, `GitHub <https://github.com/jacopotagliabue>`__
* Federico Bianchi - `LinkedIn <https://www.linkedin.com/in/federico-bianchi-3b7998121/>`__, `GitHub <https://github.com/vinid>`__
* Chloe He - `LinkedIn <https://www.linkedin.com/in/chloe-he//>`__, `GitHub <https://github.com/chloeh13q>`__
* Brian Ko - `LinkedIn <https://www.linkedin.com/in/briankosw/>`__, `GitHub <https://github.com/briankosw>`__

If you have questions or feedback, please reach out to: :code:`jacopo dot tagliabue at tooso dot ai`.

License and Citation
--------------------

All the code is released under an open MIT license. If you found *RecList* useful, or you are using it to benchmark/debug your model, please cite our pre-print (forhtcoming):

.. code-block:: bash

 @inproceedings{Chia2021BeyondNB,
   title={Beyond NDCG: behavioral testing of recommender systems with RecList},
   author={Patrick John Chia and Jacopo Tagliabue and Federico Bianchi and Chloe He and Brian Ko},
   year={2021}
 }

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

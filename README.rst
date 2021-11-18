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

*RecList* ships with some popular datasets and ready-made behavioral tests: check the paper (forthcoming) 
for more details on the relevant literature and the philosophical motivations behind the project.

If you are not familiar with the library, we suggest first taking our small [tour](#a-guided-tour) 
to get acquainted with the main abstractions through ready-made models and public datasets.

Project updates
~~~~~~~~~~~~~~~

*Nov. 2021*: the library is currently in alpha. We welcome contributions and feedback, but please be advised that the package 
may change substantially in the upcoming months. A pre-print of the companion paper is planned to be released before Dec. 2021

As the project is in active development, come back often for updates.

Summary
~~~~~~~

This doc is structured as follows:

* `Quick Start`_
* `A Guided Tour`_
* `Roadmap`_
* `Acknowledgments`_
* `License and Citation`_

Quick Start
-----------

If you want to see *RecList* in action, clone the repository, create and activate a virtual env, and install
the required packages from root. Executing `examples/coveo_complementary_rec.py` will download a 
public e-commerce dataset, train a machine learning model on it, and 
use a pre-made suite of behavioral tests to show a typical run.

.. code-block::
        python3 -m venv venv
        source venv/bin/activate
        pip install -e .
        python examples/coveo_complementary_rec.py

Running *your* model on one of the supported dataset, leveraging the pre-made tests, is as easy as implementing
a simple interface, *RecModel*.

Once you've run successfully the sample script, take the quick guided tour (coming soon) below to learn more about
the abstractions and the out-of-the-box capabilities of *RecList*.

A Guided Tour
-------------

TBC

Capabilities
------------

TBC

Datasets
~~~~~~~~

RecList features convenient wrappers around popular datasets, to help test models over known benchmarks 
in a standardized way. 

* `Coveo Data Challenge <https://github.com/coveooss/SIGIR-ecom-data-challenge>`__
* `The Million Playlist Dataset <https://engineering.atspotify.com/2018/05/30/introducing-the-million-playlist-dataset-and-recsys-challenge-2018/`__ (*coming soon*)
* `MovieLens <https://grouplens.org/datasets/movielens/>`__ (*coming soon*)

Behavioral Tests
~~~~~~~~~~~~~~~~

TBC

Roadmap
-------

TBC

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

.. code-block::

    @article{recListPre2021,
      title={Beyond NDCG: behavioral testing of recommender systems with RecList},
      author={Patrick John Chia and Jacopo Tagliabue and Federico Bianchi and Chloe He and Brian Ko},
      journal={ArXiv},
      year={forthcoming}
    }

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

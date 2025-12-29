Installation
============

Installation Methods
--------------------

From PyPI
~~~~~~~~~

.. code-block:: bash

   pip install orgnet

From Source
~~~~~~~~~~~

.. code-block:: bash

   git clone https://github.com/kylejones200/orgnet.git
   cd orgnet
   pip install -e .

Requirements
------------

Core Dependencies
~~~~~~~~~~~~~~~~~~

- Python 3.11+
- numpy >= 1.24.0
- pandas >= 2.0.0
- networkx >= 3.1
- scipy >= 1.10.0
- scikit-learn >= 1.3.0

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~

Machine Learning
^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install orgnet[ml]

Includes: PyTorch, PyTorch Geometric, Transformers, Sentence Transformers

NLP
^^^

.. code-block:: bash

   pip install orgnet[nlp]

Includes: BERTopic, Gensim

Visualization
^^^^^^^^^^^^^

.. code-block:: bash

   pip install orgnet[viz]

Includes: Matplotlib, Seaborn, Plotly, Pyvis, Dash

API
^^^

.. code-block:: bash

   pip install orgnet[api]

Includes: Flask, Flask-CORS

Temporal Analysis
^^^^^^^^^^^^^^^^^

.. code-block:: bash

   pip install orgnet[temporal]

Includes: Ruptures

Development
^^^^^^^^^^^

.. code-block:: bash

   pip install orgnet[dev]

Includes: pytest, pytest-cov, black, flake8, faker, jupyter

Post-Installation
------------------

Download spaCy Language Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python -m spacy download en_core_web_sm

This is required for NLP features.


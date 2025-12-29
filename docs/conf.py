"""Configuration file for the Sphinx documentation builder."""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.abspath(".."))

# Project information
project = "orgnet"
copyright = "2024, Kyle Jones"
author = "Kyle Jones"
release = "0.1.0"

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# HTML output options
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_theme_options = {
    "collapse_navigation": False,
    "display_version": True,
    "logo_only": False,
}

# Autodoc settings
autodoc_mock_imports = [
    "torch",
    "torch_geometric",
    "transformers",
    "sentence_transformers",
    "bertopic",
    "gensim",
    "matplotlib",
    "seaborn",
    "plotly",
    "pyvis",
    "dash",
    "dash_bootstrap_components",
    "ruptures",
    "flask",
    "flask_cors",
    "python_igraph",
    "node2vec",
    "spacy",
    "nltk",
]

# Napoleon settings for Google/NumPy style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}


# Configuration file for the Sphinx documentation builder.

import os
import sys
from importlib.metadata import PackageNotFoundError, version as pkg_version
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------
project = 'trspecfit'
copyright = '2025, The Regents of the University of California'
author = 'Johannes Mahl'
try:
    release = pkg_version("trspecfit")
except PackageNotFoundError:
    release = "0.0.0"

#version = ".".join(release.split(".")[:2])  # optional short X.Y
version = release  # full version

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',        # Auto-generate docs from docstrings
    'sphinx.ext.napoleon',       # Support NumPy/Google style docstrings
    'sphinx.ext.viewcode',       # Add links to source code
    'sphinx.ext.mathjax',        # Math equations
    'sphinx.ext.intersphinx',    # Link to other projects' docs
    'myst_parser',               # Markdown support
    'nbsphinx',                  # Jupyter notebook support
]

# Support both .rst and .md files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# Napoleon settings for NumPy-style docstrings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx mapping (link to other docs)
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'lmfit': ('https://lmfit.github.io/lmfit-py/', None),
}

# NBSphinx settings
nbsphinx_execute = 'never'  # Don't execute notebooks when building docs

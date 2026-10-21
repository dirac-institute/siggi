"""Sphinx configuration for the siggi documentation."""

import os
import sys
from importlib.metadata import version as _get_version

# Make the package importable for autodoc when the docs are built from a
# source checkout without an editable install (e.g. a local ``sphinx-build``).
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "siggi"
author = "Bryce Kalmbach"
copyright = "2018, J. Bryce Kalmbach"

try:
    release = _get_version("siggi")
except Exception:
    release = "0.1.5"
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

# Parse both Markdown (MyST) and reStructuredText sources.
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

master_doc = "index"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Autodoc / Napoleon ------------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
napoleon_numpy_docstring = True
napoleon_google_docstring = False

# -- MyST --------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
]

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

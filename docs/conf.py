# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import importlib.metadata

# -- Project information -----------------------------------------------------
project = "OncoPrep"
copyright = "2024–2026, Nikitas C. Koussis"
author = "Nikitas C. Koussis"
release = importlib.metadata.version("oncoprep")

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Autodoc settings --------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_mock_imports = [
    "nipype",
    "niworkflows",
    "nibabel",
    "pybids",
    "bids",
    "antspy",
    "antspyx",
    "ants",
    "templateflow",
    "torch",
    "torchvision",
    "torchaudio",
    "radiomics",
    "pyradiomics",
    "acres",
    "nilearn",
    "pydicom",
    "weasyprint",
    "dcm2niix",
    "picsl_greedy",
    "freesurfer",
    "hd_bet",
    "HD_BET",
]

autosummary_generate = False

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- MyST (Markdown) settings ------------------------------------------------
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# -- Intersphinx mapping -----------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "nibabel": ("https://nipy.org/nibabel/", None),
    "nipype": ("https://nipype.readthedocs.io/en/latest/", None),
}

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "OncoPrep"
html_static_path = ["_static"]
html_css_files = []

# Furo theme options
html_theme_options = {
    "source_repository": "https://github.com/nikitas-k/oncoprep",
    "source_branch": "main",
    "source_directory": "docs/",
}

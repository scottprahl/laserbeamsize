"""
Sphinx configuration for laserbeamsize documentation.

Uses:
- sphinx.ext.napoleon for Google-style docstrings
- nbsphinx for rendering Jupyter notebooks (pre-executed; no execution on RTD)
"""

from importlib.metadata import version as pkg_version

project = "laserbeamsize"
release = pkg_version(project)
version = release

root_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_automodapi.automodapi",
    "nbsphinx",
]

napoleon_use_param = False
napoleon_use_rtype = False
numpydoc_show_class_members = False

exclude_patterns = [
    "_build",
    ".ipynb_checkpoints",
    "images/readme_images.ipynb",
]

nbsphinx_execute = "never"
nbsphinx_allow_errors = False

html_theme = "sphinx_rtd_theme"
html_scaled_image_link = False
html_sourcelink_suffix = ""

html_static_path = ["_static"]
html_css_files = ["custom.css"]

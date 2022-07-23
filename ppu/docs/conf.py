# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os.path

# -- Project information -----------------------------------------------------

project = "PPU"
copyright = "2021, PPU authors"
author = "PPU authors"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.graphviz",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.extlinks",
    "sphinx.ext.autosectionlabel",
    "myst_parser",
    "sphinxcontrib.actdiag",
    "sphinxcontrib.blockdiag",
    "sphinxcontrib.mermaid",
    "sphinxcontrib.nwdiag",
    "sphinxcontrib.packetdiag",
    "sphinxcontrib.rackdiag",
    "sphinxcontrib.seqdiag",
    "sphinx_markdown_tables",
]

# Make sure the target is unique
autosectionlabel_prefix_document = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
# html_theme_options = {'page_width': 'max-content'}
# html_theme = 'classic'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# note: 'bysource' does not work for c++ extensions
autodoc_member_order = "groupwise"

# Enable TODO
todo_include_todos = True

# config blockdiag

# global variables
extlinks = {
    "ppu_doc_host": ("https://ppu.readthedocs.io/zh/latest", "doc "),
    "ppu_code_host": ("https://github.com/secretflow", "code "),
}

font_file = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
if os.path.isfile(font_file):
    blockdiag_fontpath = font_file
    seqdiag_fontpath = font_file


# app setup hook
def setup(app):
    app.add_config_value(
        "recommonmark_config",
        {
            # 'url_resolver': lambda url: github_doc_root + url,
            "auto_toc_tree_section": "Contents",
        },
        True,
    )

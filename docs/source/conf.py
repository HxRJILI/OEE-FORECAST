# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'OEE-Forecasting'
copyright = '2025, RJILI HOUSSAM, WIAME EL HAFID'
author = 'RJILI HOUSSAM, WIAME EL HAFID'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add the extensions we need
extensions = [
    'sphinx.ext.autodoc',        # For automatic API documentation
    'sphinx.ext.viewcode',       # Add links to view the source code
    'sphinx.ext.napoleon',       # Support for NumPy and Google style docstrings
    'sphinx.ext.mathjax',        # For math equations
    'recommonmark',              # Support for markdown files
    'nbsphinx',                  # Support for Jupyter notebooks (install with pip if needed)
]

# Support for markdown files
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Change to the Read the Docs theme
html_theme = 'sphinx_rtd_theme'  # This is the Read the Docs theme
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'display_version': True,
    'logo_only': False,
}

# -- Options for autodoc extension ------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#configuration

# Sort members by type
autodoc_member_order = 'groupwise'

# -- Options for Napoleon extension -----------------------------------------
# https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
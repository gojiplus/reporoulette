# Add these to your conf.py
import os
import sys
sys.path.insert(0, os.path.abspath('..'))  # Add your project root

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
]

html_theme = 'sphinx_rtd_theme'
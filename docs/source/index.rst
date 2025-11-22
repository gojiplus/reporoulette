.. reporoulette documentation master file, created by
   sphinx-quickstart on Wed Mar 26 20:52:45 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

RepoRoulette: Randomly Sample GitHub Repositories
==================================================

.. automodule:: reporoulette
   :members:
   :undoc-members:
   :show-inheritance:

RepoRoulette provides multiple methods for randomly sampling GitHub repositories:

* **ID-based sampling**: Probes random repository IDs
* **Temporal sampling**: Weighted sampling based on repository activity by time period
* **BigQuery sampling**: Advanced querying using Google BigQuery's GitHub dataset
* **GitHub Archive sampling**: Event-based sampling from GitHub Archive files

Quick Start
-----------

.. code-block:: python

   from reporoulette import sample

   # Sample 10 repositories using temporal sampling
   results = sample(method='temporal', n_samples=10)
   print(f"Found {len(results['samples'])} repositories")

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   samplers

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

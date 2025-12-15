BigQuery Sampler
================

The BigQuery sampler leverages Google BigQuery's public GitHub dataset to sample repositories with advanced filtering capabilities.

.. autoclass:: reporoulette.BigQuerySampler
   :members:
   :undoc-members:
   :show-inheritance:

Advantages
----------

- Handles large sample sizes efficiently
- Powerful filtering and stratification options
- Not limited by GitHub API rate limits
- Access to historical data and metadata

Disadvantages
-------------

- Requires Google Cloud Platform account
- Can be expensive for large queries
- Dataset may have slight delays (24-48 hours)

Usage Example
-------------

.. code-block:: python

   from reporoulette import BigQuerySampler

   # Direct usage
   sampler = BigQuerySampler(
       credentials_path="/path/to/credentials.json",
       project_id="your-gcp-project"
   )
   repos = sampler.sample(n_samples=100)

   # Using convenience function
   from reporoulette import sample
   results = sample(
       method='bigquery',
       n_samples=100,
       credentials_path="/path/to/credentials.json",
       project_id="your-gcp-project"
   )

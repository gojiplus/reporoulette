Sample Function
===============

The ``sample()`` function provides a convenient unified interface to all sampling methods without needing to instantiate sampler classes directly.

.. autofunction:: reporoulette.sample

Usage Examples
--------------

The sample function automatically handles sampler instantiation and configuration:

.. code-block:: python

   from reporoulette import sample

   # ID-based sampling
   results = sample(method='id', n_samples=10)

   # Temporal sampling (default method)
   results = sample(n_samples=20)

   # BigQuery sampling with credentials
   results = sample(
       method='bigquery',
       n_samples=100,
       credentials_path="/path/to/credentials.json",
       project_id="your-gcp-project"
   )

   # GitHub Archive sampling
   results = sample(method='archive', n_samples=50)

Return Format
-------------

All methods return a dictionary with:

- ``method``: The sampling method used
- ``params``: Parameters passed to the sampler
- ``attempts``: Total sampling attempts made
- ``success_rate``: Ratio of successful to total attempts
- ``samples``: List of repository data dictionaries

.. code-block:: python

   {
       "method": "temporal",
       "params": {"start_date": "2024-01-01", ...},
       "attempts": 25,
       "success_rate": 0.8,
       "samples": [
           {"full_name": "user/repo1", "stars": 42, ...},
           {"full_name": "user/repo2", "stars": 123, ...},
           ...
       ]
   }

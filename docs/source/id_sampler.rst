ID Sampler
==========

The ID-based sampler uses GitHub's sequential repository ID system to generate truly random samples by probing random IDs from the valid ID range.

.. autoclass:: reporoulette.IDSampler
   :members:
   :undoc-members:
   :show-inheritance:

Advantages
----------

- Truly random sampling across all public repositories
- Simple and straightforward approach
- Good for unbiased statistical sampling

Disadvantages
-------------

- Low hit rate due to many invalid IDs (private/deleted repos)
- Any filtering must be done after sampling
- Limited by GitHub API rate limits

Usage Example
-------------

.. code-block:: python

   from reporoulette import IDSampler

   # Direct usage
   sampler = IDSampler(token="your_github_token")
   repos = sampler.sample(n_samples=10)

   # Using convenience function
   from reporoulette import sample
   results = sample(method='id', n_samples=10)

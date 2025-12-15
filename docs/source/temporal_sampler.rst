Temporal Sampler
================

The temporal sampler randomly selects days within a specified date range and retrieves repositories updated during those periods using weighted sampling based on repository activity.

.. autoclass:: reporoulette.TemporalSampler
   :members:
   :undoc-members:
   :show-inheritance:

Advantages
----------

- Can filter repositories during sampling
- Weighted approach provides more active repositories
- Customizable date ranges

Disadvantages
-------------

- May be biased toward more active repositories
- Limited by GitHub API rate limits
- Requires careful parameter tuning

Usage Example
-------------

.. code-block:: python

   from reporoulette import TemporalSampler
   from datetime import datetime, timedelta

   # Direct usage
   sampler = TemporalSampler(token="your_github_token")
   repos = sampler.sample(
       n_samples=10,
       start_date=datetime.now() - timedelta(days=365),
       end_date=datetime.now()
   )

   # Using convenience function
   from reporoulette import sample
   results = sample(method='temporal', n_samples=10)

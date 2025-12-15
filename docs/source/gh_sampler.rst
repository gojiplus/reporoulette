GitHub Archive Sampler
======================

The GitHub Archive sampler fetches repositories by sampling events from GitHub Archive, which records the public GitHub timeline.

.. autoclass:: reporoulette.GHArchiveSampler
   :members:
   :undoc-members:
   :show-inheritance:

Advantages
----------

- Free to use (no API tokens required)
- Access to event-based data
- Can sample based on specific event types

Disadvantages
-------------

- Limited to repositories with recent activity
- May be slower due to processing compressed archives
- Less control over sampling criteria

Usage Example
-------------

.. code-block:: python

   from reporoulette import GHArchiveSampler
   from datetime import datetime

   # Direct usage
   sampler = GHArchiveSampler()
   repos = sampler.sample(
       n_samples=10,
       date=datetime(2024, 1, 15),
       hour=12
   )

   # Using convenience function
   from reporoulette import sample
   results = sample(method='archive', n_samples=10)

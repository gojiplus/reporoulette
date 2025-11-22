Sampling Methods
================

RepoRoulette provides four distinct methods for randomly sampling GitHub repositories, each with its own advantages and use cases.

ID-Based Sampling
-----------------

.. autoclass:: reporoulette.IDSampler
   :members:
   :undoc-members:
   :show-inheritance:

The ID-based sampler uses GitHub's sequential repository ID system to generate truly random samples by probing random IDs from the valid ID range.

**Advantages:**
- Truly random sampling across all public repositories
- Simple and straightforward approach
- Good for unbiased statistical sampling

**Disadvantages:**
- Low hit rate due to many invalid IDs (private/deleted repos)
- Any filtering must be done after sampling
- Limited by GitHub API rate limits

Temporal Sampling
-----------------

.. autoclass:: reporoulette.TemporalSampler
   :members:
   :undoc-members:
   :show-inheritance:

The temporal sampler randomly selects days within a specified date range and retrieves repositories updated during those periods using weighted sampling based on repository activity.

**Advantages:**
- Can filter repositories during sampling
- Weighted approach provides more active repositories
- Customizable date ranges

**Disadvantages:**
- May be biased toward more active repositories
- Limited by GitHub API rate limits
- Requires careful parameter tuning

BigQuery Sampling
-----------------

.. autoclass:: reporoulette.BigQuerySampler
   :members:
   :undoc-members:
   :show-inheritance:

The BigQuery sampler leverages Google BigQuery's public GitHub dataset to sample repositories with advanced filtering capabilities.

**Advantages:**
- Handles large sample sizes efficiently
- Powerful filtering and stratification options
- Not limited by GitHub API rate limits
- Access to historical data and metadata

**Disadvantages:**
- Requires Google Cloud Platform account
- Can be expensive for large queries
- Dataset may have slight delays (24-48 hours)

GitHub Archive Sampling
------------------------

.. autoclass:: reporoulette.GHArchiveSampler
   :members:
   :undoc-members:
   :show-inheritance:

The GitHub Archive sampler fetches repositories by sampling events from GitHub Archive, which records the public GitHub timeline.

**Advantages:**
- Free to use (no API tokens required)
- Access to event-based data
- Can sample based on specific event types

**Disadvantages:**
- Limited to repositories with recent activity
- May be slower due to processing compressed archives
- Less control over sampling criteria

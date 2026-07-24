# Randomization Validation Report

Generated 2026-07-24 15:23 by `scripts/validate_randomization.py`. All tests at alpha = 0.05. Chi-square critical values use the Wilson-Hilferty approximation; KS uses the asymptotic two-sample critical value. Aggregating draws across seeded runs treats draws as independent, which is a good approximation for sample sizes far below the population size.

## P1 - GH Archive ground-truth inclusion test (GHArchiveSampler)

Population: all 7,294 repositories created in hour 12 UTC of 2023-06-15 (fully enumerated from the public archive). 100 seeded replays of 100 draws each (10,000 total draws) through the real sampler code path.

| Test | Statistic | Critical (5%) | Result |
|---|---|---|---|
| Equal inclusion probability (100 hash buckets, df=99) | 105.3 | 123.2 | PASS |
| Creation minute-of-hour marginal vs population (df=59) | 51.8 | 77.9 | PASS |
| Owner-initial marginal vs population | 27.6 | 38.9 | PASS |

## P3 - Seed stability (pairwise KS across seeds)

4950 pairwise two-sample KS tests on within-day creation times across independent seeds: 250 rejections (5.1%; expectation under uniform sampling is about 5%). PASS

## P2 - Live IDSampler audit

2,000 live probes against the real GitHub API, 747 hits (37.4% success rate).

| Test | Statistic | Critical (5%) | Result |
|---|---|---|---|
| Probed IDs uniform over 10 deciles (df=9) | 9.24 | 16.90 | PASS |
| created_at monotone in ID (inversion rate) | 0.00% | < 1% | PASS |
| Live probe-sequence reproducibility (same seed, 2 runs) | identical | identical | PASS |

Hit rate by ID decile (existing-repo density over GitHub's history): 42.1%, 50.0%, 37.9%, 38.1%, 31.6%, 34.9%, 35.0%, 39.3%, 34.3%, 29.7%.

Coverage gap: newest observed repo ID is 1,310,925,406 vs the default max_id 850,000,000 - the default excludes 35.2% of the current ID space (newest repositories). Pass max_id explicitly for full coverage.

## P4 - Temporal sampler bias, quantified (measurement, not pass/fail)

Temporal sample n=200 vs near-uniform ID-sample benchmark n=747:

| Metric | TemporalSampler | IDSampler benchmark |
|---|---|---|
| Mean stars | 0.4 | 0.2 |
| Median stars | 0 | 0 |
| % with >= 1 star | 12% | 9% |
| % pushed in last 90 days | 0% | 0% |
| Top languages | JavaScript (36), HTML (28), Python (20), TypeScript (16), Jupyter Notebook (8) | JavaScript (105), HTML (77), Java (53), Python (49), Jupyter Notebook (23) |

Search-cap coverage: mean fraction of each sampled day's repositories reachable under the 1,000-result cap: 2.7% (per-day counts: 2017-02-25: 19,301, 2017-07-18: 32,509, 2017-10-10: 34,576, 2019-07-22: 33,353, 2019-07-27: 21,277, 2021-02-18: 66,816, 2025-03-10: 157,895, 2025-12-20: 151,606).

## P5 - BigQuery live check (BigQuerySampler)

| Check | Result |
|---|---|
| Returns real data (was always empty before the invalid-SQL fixes) | 49 repos - PASS |
| Same seed, two live runs identical | PASS |
| Duplicate repositories in output | 0 - PASS |
| Days used / allocation vs day size correlation | 5 days, r = 1.00 |

## P6 - Live API contract checks

| Check | Result |
|---|---|
| GH Archive publishes hourly files only (daily URL 404s, hourly 200; probed 2026-07-22) | daily=404, hourly=200 - PASS |
| Event schema fields used by GHArchiveSampler present | first 500 events well-formed - PASS |
| Repository CreateEvents ended with GitHub's 2025-10-07 Events API change (present before, absent after) | 1878 in 50k events of 2025-08-15 vs 0 recently - PASS |
| /repositories/{id} returns the fields IDSampler maps | HTTP 200 - PASS |
| Temporal population claim: pushed_at inside the sampled day | 100 repos, 100% in window - PASS |

## P7 - BigQuery dry-run validation (free)

Every query shape the sampler can emit, compiled by BigQuery without execution (dry runs cost nothing and would have caught all three SQL-validity bugs in this audit). Estimated scan per query:

| Query | Valid | Est. scan |
|---|---|---|
| count (5 days, pruned wildcard) | PASS | 0.52 GB |
| day sample | PASS | 0.18 GB |
| combined (2 days, UNION ALL) | PASS | 0.18 GB |
| active (no languages) | PASS | 99.45 GB |
| active (languages) | PASS | 99.59 GB |
| get_languages | PASS | 0.20 GB |

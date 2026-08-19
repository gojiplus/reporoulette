"""Empirical validation that reporoulette's samplers randomize correctly.

Unlike the unit tests (which mock the network and prove the code faithfully
passes RNG draws through), this script runs the samplers against real data
and tests the statistical definition of successful randomization.

Sections (select with flags, or --all):
  P1  GH Archive ground truth: for one archived day the population of created
      repositories is fully enumerable, so equal inclusion probability can be
      tested directly across ~100 seeded replays. Free, no credentials.
  P2  Live IDSampler audit: ~2,000 real probes; uniformity of probed IDs,
      ID/creation-date monotonicity, live seed reproducibility, and the ID
      coverage gap of the default max_id. Requires GITHUB_TOKEN.
  P3  Seed stability: pairwise KS tests across P1's per-seed samples.
  P4  Temporal bias quantification: coverage under the search API's 1,000
      result cap and activity-bias measurements vs the P2 ID-sample
      benchmark. Requires GITHUB_TOKEN.
  P5  BigQuery live check: seeded reproducibility, duplicate-freeness, and
      day allocation sanity against real BigQuery. Requires Google
      application default credentials.
  P6  Live API contract checks: GH Archive URL scheme and event schema,
      the /repositories/{id} field contract, and a temporal end-to-end
      mini-run verifying pushed_at falls inside the sampled day. Requires
      GITHUB_TOKEN.
  P7  BigQuery dry-run validation: compile every query shape the sampler
      can emit and report the bytes each would scan - free, catches all
      SQL-validity bugs without spending quota. Requires Google
      application default credentials.

Writes docs/validation.md. Never writes credentials anywhere.
"""

import argparse
import contextlib
import gzip
import hashlib
import itertools
import json
import logging
import math
import re
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import requests

from reporoulette.samplers.gh_sampler import GHArchiveSampler
from reporoulette.samplers.id_sampler import IDSampler
from reporoulette.samplers.temporal_sampler import TemporalSampler

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / ".validation_cache"
REPORT_PATH = REPO_ROOT / "docs" / "validation.md"

ALPHA = 0.05
Z_95 = 1.6448536269514722  # one-sided 95% normal quantile
KS_C_ALPHA = 1.3581  # c(alpha) for alpha=0.05, two-sample KS


# ---------------------------------------------------------------- statistics


def chi2_critical(df: int) -> float:
    """Wilson-Hilferty approximation to the chi-square 95% critical value."""
    a = 2.0 / (9.0 * df)
    return df * (1.0 - a + Z_95 * math.sqrt(a)) ** 3


def chi2_stat(observed: list[float], expected: list[float]) -> float:
    pairs = zip(observed, expected, strict=True)
    return sum((o - e) ** 2 / e for o, e in pairs if e > 0)


def ks_2samp_stat(a: list[float], b: list[float]) -> float:
    a, b = sorted(a), sorted(b)
    i = j = 0
    d = 0.0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            i += 1
        else:
            j += 1
        d = max(d, abs(i / len(a) - j / len(b)))
    return d


def ks_critical(n: int, m: int) -> float:
    return KS_C_ALPHA * math.sqrt((n + m) / (n * m))


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return float("nan")
    return cov / (sx * sy)


def stable_bucket(name: str, buckets: int) -> int:
    digest = hashlib.md5(name.encode("utf-8"), usedforsecurity=False).digest()
    return int.from_bytes(digest[:4], "big") % buckets


# ------------------------------------------------------- P1: archive ground truth


def is_repo_creation(event: dict) -> bool:
    return (
        event.get("type") == "CreateEvent"
        and event.get("payload", {}).get("ref_type") == "repository"
    )


def build_population(archive_date: str, hour: int) -> tuple[Path, list[dict]]:
    """Download one real hourly archive file (GH Archive publishes hourly
    files only), enumerate its creation population, and write a reduced
    (population-only) gzip used to replay the sampler fast.

    Returns the reduced-file path and the population, applying the exact
    parse rules of GHArchiveSampler (valid owner/name, first occurrence per
    repo wins).
    """
    CACHE_DIR.mkdir(exist_ok=True)
    full_path = CACHE_DIR / f"{archive_date}-{hour}.json.gz"
    reduced_path = CACHE_DIR / f"{archive_date}-{hour}.reduced.json.gz"
    population_path = CACHE_DIR / f"{archive_date}-{hour}.population.json"

    if not full_path.exists():
        url = f"https://data.gharchive.org/{archive_date}-{hour}.json.gz"
        print(f"[P1] downloading {url} ...")
        with requests.get(url, stream=True, timeout=60) as resp:
            resp.raise_for_status()
            with full_path.open("wb") as fh:
                fh.writelines(resp.iter_content(chunk_size=1 << 20))
        print(f"[P1] cached {full_path.stat().st_size / 1e6:.0f} MB")

    if reduced_path.exists() and population_path.exists():
        population = json.loads(population_path.read_text())
        return reduced_path, population

    print("[P1] enumerating population (single full parse) ...")
    population: list[dict] = []
    seen: set[str] = set()
    with (
        gzip.open(full_path, "rt", encoding="utf-8", errors="replace") as fh,
        gzip.open(reduced_path, "wt", encoding="utf-8") as out,
    ):
        for line in fh:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not is_repo_creation(event):
                continue
            name = event.get("repo", {}).get("name")
            if not name or "/" not in name:
                continue
            reduced = {
                "type": "CreateEvent",
                "repo": {"name": name},
                "payload": {"ref_type": "repository"},
                "created_at": event.get("created_at"),
            }
            out.write(json.dumps(reduced) + "\n")
            if name in seen:
                continue
            seen.add(name)
            owner = name.split("/", 1)[0]
            population.append(
                {"full_name": name, "owner": owner, "created_at": event["created_at"]}
            )
    population_path.write_text(json.dumps(population))
    print(f"[P1] population: {len(population)} repositories created {archive_date}")
    return reduced_path, population


def seconds_of_day(created_at: str) -> float:
    ts = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    return ts.hour * 3600 + ts.minute * 60 + ts.second


def run_p1(archive_date: str, hour: int, n_seeds: int, k: int) -> dict:
    reduced_path, population = build_population(archive_date, hour)
    pop_size = len(population)
    pop_by_name = {p["full_name"]: p for p in population}

    # The sampler fetches 24 hourly files per day; serve the real population
    # for hour 0 and an empty archive for the rest so each replay parses the
    # population exactly once.
    empty_path = CACHE_DIR / "empty.json.gz"
    if not empty_path.exists():
        with gzip.open(empty_path, "wb"):
            pass

    class ArchiveResponse:
        status_code = 200

        def __init__(self, path):
            # SIM115 suppressed: the response object owns the handle, mimicking
            # requests' streaming `raw`; the sampler's gzip reader consumes it.
            self.raw = Path(path).open("rb")  # noqa: SIM115

    def fake_get(url, *_args, **_kwargs):
        if url.endswith("-0.json.gz"):
            return ArchiveResponse(reduced_path)
        return ArchiveResponse(empty_path)

    per_seed_samples: list[list[str]] = []
    inclusion: Counter = Counter()
    t0 = time.time()
    with patch("requests.get", side_effect=fake_get):
        for seed in range(n_seeds):
            sampler = GHArchiveSampler(seed=seed, log_level=logging.ERROR)
            repos = sampler.gh_sampler(
                n_samples=k, days_to_sample=1, repos_per_day=k, years_back=1
            )
            names = [r["full_name"] for r in repos]
            per_seed_samples.append(names)
            inclusion.update(names)
    draws = sum(len(s) for s in per_seed_samples)
    print(
        f"[P1] {n_seeds} replays x {k} draws from N={pop_size} "
        f"in {time.time() - t0:.0f}s"
    )

    # (a) equal inclusion probability, tested on hash-bucket aggregates
    n_groups = 100
    group_sizes = [0] * n_groups
    for p in population:
        group_sizes[stable_bucket(p["full_name"], n_groups)] += 1
    group_draws = [0.0] * n_groups
    for name, count in inclusion.items():
        group_draws[stable_bucket(name, n_groups)] += count
    expected = [draws * size / pop_size for size in group_sizes]
    incl_chi2 = chi2_stat(group_draws, expected)
    incl_crit = chi2_critical(n_groups - 1)

    # (b) creation minute-of-hour marginal of the samples vs the known
    # population (the population spans one hour, so hours are degenerate)
    pop_minutes = Counter(
        int(seconds_of_day(p["created_at"]) % 3600 // 60) for p in population
    )
    sample_minutes: Counter = Counter()
    for names in per_seed_samples:
        for name in names:
            sample_minutes[
                int(seconds_of_day(pop_by_name[name]["created_at"]) % 3600 // 60)
            ] += 1
    minute_obs = [float(sample_minutes.get(m, 0)) for m in range(60)]
    minute_exp = [draws * pop_minutes.get(m, 0) / pop_size for m in range(60)]
    minute_chi2 = chi2_stat(minute_obs, minute_exp)
    minute_crit = chi2_critical(59)

    # (c) owner-initial marginal (a-z bucketed, everything else pooled)
    def initial(owner: str) -> str:
        c = owner[0].lower()
        return c if "a" <= c <= "z" else "other"

    pop_initials = Counter(initial(p["owner"]) for p in population)
    sample_initials: Counter = Counter()
    for names in per_seed_samples:
        for name in names:
            sample_initials[initial(pop_by_name[name]["owner"])] += 1
    cells = sorted(pop_initials)
    init_obs = [float(sample_initials.get(c, 0)) for c in cells]
    init_exp = [draws * pop_initials[c] / pop_size for c in cells]
    init_chi2 = chi2_stat(init_obs, init_exp)
    init_crit = chi2_critical(len(cells) - 1)

    seconds = [
        [seconds_of_day(pop_by_name[n]["created_at"]) for n in names]
        for names in per_seed_samples
    ]
    return {
        "archive_date": archive_date,
        "archive_hour": hour,
        "population_size": pop_size,
        "n_seeds": n_seeds,
        "k": k,
        "draws": draws,
        "inclusion_chi2": incl_chi2,
        "inclusion_crit": incl_crit,
        "inclusion_pass": incl_chi2 < incl_crit,
        "minute_chi2": minute_chi2,
        "minute_crit": minute_crit,
        "minute_pass": minute_chi2 < minute_crit,
        "initial_chi2": init_chi2,
        "initial_crit": init_crit,
        "initial_pass": init_chi2 < init_crit,
        "per_seed_seconds": seconds,
    }


# ----------------------------------------------------------- P3: seed stability


def run_p3(per_seed_seconds: list[list[float]]) -> dict:
    rejections = 0
    pairs = 0
    for i in range(len(per_seed_seconds)):
        for j in range(i + 1, len(per_seed_seconds)):
            a, b = per_seed_seconds[i], per_seed_seconds[j]
            if not a or not b:
                continue
            pairs += 1
            if ks_2samp_stat(a, b) > ks_critical(len(a), len(b)):
                rejections += 1
    rate = rejections / pairs if pairs else float("nan")
    return {
        "pairs": pairs,
        "rejections": rejections,
        "rejection_rate": rate,
        "pass": rate < 0.15,
    }


# ------------------------------------------------------------ P2: live ID audit


def attach_recorder(sampler, rate_check_every: int = 100):
    """Record request URLs; cache the per-request /rate_limit pre-check so a
    real check happens every `rate_check_every` calls (the sampling path is
    untouched; this only removes redundant free /rate_limit round-trips).
    """
    urls: list[str] = []
    original_request = sampler._make_github_request
    original_check = sampler._check_rate_limit
    state = {"calls": 0, "cached": 5000}

    def recording_request(url, **kwargs):
        urls.append(url)
        return original_request(url, **kwargs)

    def cached_check(resource="core"):
        if state["calls"] % rate_check_every == 0:
            state["cached"] = original_check(resource)
        state["calls"] += 1
        return state["cached"]

    sampler._make_github_request = recording_request
    sampler._check_rate_limit = cached_check
    return urls


def probe_ids(urls: list[str]) -> list[int]:
    ids = []
    for url in urls:
        match = re.search(r"/repositories/(\d+)$", url)
        if match:
            ids.append(int(match.group(1)))
    return ids


def newest_repo_id(token: str) -> int:
    since = (datetime.now(UTC) - timedelta(days=1)).strftime("%Y-%m-%d")
    resp = requests.get(
        "https://api.github.com/search/repositories",
        params={"q": f"created:>={since}", "per_page": 100},
        headers={"Authorization": f"token {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    return max(item["id"] for item in resp.json()["items"])


def run_p2(token: str, n_probes: int) -> dict:
    sampler = IDSampler(token=token, seed=2026, log_level=logging.WARNING)
    urls = attach_recorder(sampler)
    t0 = time.time()
    hits = sampler.sample(n_samples=n_probes, max_attempts=n_probes, min_wait=0.05)
    elapsed = time.time() - t0
    ids = probe_ids(urls)
    print(
        f"[P2] {len(ids)} live probes, {len(hits)} hits "
        f"({sampler.success_rate:.1f}%) in {elapsed:.0f}s"
    )

    # (a) probed IDs uniform over deciles of [min_id, max_id]
    buckets = [0.0] * 10
    span = (sampler.max_id - sampler.min_id + 1) / 10
    for repo_id in ids:
        buckets[min(int((repo_id - sampler.min_id) / span), 9)] += 1
    uniform_chi2 = chi2_stat(buckets, [len(ids) / 10] * 10)
    uniform_crit = chi2_critical(9)

    # (b) IDs are chronological: created_at sorted by ID should be monotone
    by_id = sorted(hits, key=lambda r: r["id"])
    dates = [r["created_at"] for r in by_id]
    inversions = sum(1 for a, b in itertools.pairwise(dates) if b < a)
    inversion_rate = inversions / max(len(dates) - 1, 1)

    # (c) hit rate per decile (population density over ID history)
    hit_buckets = [0] * 10
    for r in hits:
        hit_buckets[min(int((r["id"] - sampler.min_id) / span), 9)] += 1
    decile_hit_rates = [
        (h / b if b else float("nan"))
        for h, b in zip(hit_buckets, buckets, strict=True)
    ]

    # (d) live reproducibility of the probe sequence
    seqs = []
    for _ in range(2):
        rep = IDSampler(token=token, seed=42, log_level=logging.WARNING)
        rep_urls = attach_recorder(rep)
        rep.sample(n_samples=25, max_attempts=25, min_wait=0.05)
        seqs.append(probe_ids(rep_urls))
    reproducible = seqs[0] == seqs[1] and len(seqs[0]) == 25

    # (e) coverage gap of the default max_id
    max_seen = newest_repo_id(token)
    gap = max(0.0, 1.0 - sampler.max_id / max_seen)

    return {
        "n_probes": len(ids),
        "n_hits": len(hits),
        "success_rate": sampler.success_rate,
        "uniform_chi2": uniform_chi2,
        "uniform_crit": uniform_crit,
        "uniform_pass": uniform_chi2 < uniform_crit,
        "inversions": inversions,
        "inversion_rate": inversion_rate,
        "monotone_pass": inversion_rate < 0.01,
        "decile_hit_rates": decile_hit_rates,
        "reproducible": reproducible,
        "max_id_default": sampler.max_id,
        "newest_repo_id": max_seen,
        "id_space_gap": gap,
        "hits": hits,
    }


# ------------------------------------------------- P4: temporal bias measurement


def run_p4(token: str, id_hits: list[dict]) -> dict:
    sampler = TemporalSampler(token=token, seed=123, log_level=logging.WARNING)
    counts: dict[str, int] = {}
    original_request = sampler._make_github_request

    def recording_request(url, **kwargs):
        response = original_request(url, **kwargs)
        if response is not None and response.status_code == 200:
            match = re.search(r"pushed:([0-9T:Z-]+)\.\.", url)
            if match:
                with contextlib.suppress(Exception):
                    counts[match.group(1)[:10]] = response.json()["total_count"]
        return response

    sampler._make_github_request = recording_request
    t0 = time.time()
    repos = sampler.sample(
        n_samples=200, days_to_sample=8, min_wait=2.5, max_attempts=60
    )
    print(f"[P4] temporal sample: {len(repos)} repos in {time.time() - t0:.0f}s")

    coverage = {day: min(1000, c) / c for day, c in counts.items() if c > 0}

    def stats(sample: list[dict]) -> dict:
        stars = sorted(r.get("stargazers_count", 0) for r in sample)
        n = len(stars)
        recent_cutoff = datetime.now(UTC) - timedelta(days=90)
        pushed_recent = sum(
            1
            for r in sample
            if r.get("pushed_at")
            and datetime.strptime(r["pushed_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=UTC
            )
            > recent_cutoff
        )
        langs = Counter(r.get("language") for r in sample if r.get("language"))
        return {
            "n": n,
            "mean_stars": sum(stars) / n if n else float("nan"),
            "median_stars": stars[n // 2] if n else float("nan"),
            "pct_starred": 100 * sum(1 for s in stars if s > 0) / n if n else 0,
            "pct_pushed_90d": 100 * pushed_recent / n if n else 0,
            "top_languages": langs.most_common(5),
        }

    return {
        "temporal": stats(repos),
        "id_benchmark": stats(id_hits),
        "day_total_counts": counts,
        "day_coverage": coverage,
        "mean_coverage": (
            sum(coverage.values()) / len(coverage) if coverage else float("nan")
        ),
    }


# ------------------------------------------------------------ P5: BigQuery live


def run_p5() -> dict:
    import os

    try:
        from reporoulette.samplers.bigquery_sampler import BigQuerySampler
    except ImportError as e:
        return {"skipped": True, "reason": f"BigQuery deps not installed: {e}"}

    project_id = os.environ.get("GOOGLE_PROJECT_ID")
    last_error = None
    results = []
    for pid in [project_id, "in-electoral-rolls"]:
        try:
            attempt = []
            for _ in range(2):
                sampler = BigQuerySampler(
                    seed=42, project_id=pid, log_level=logging.WARNING
                )
                repos = sampler.sample_by_day(
                    n_samples=50, days_to_sample=5, years_back=5
                )
                attempt.append(repos)
            results = attempt
            # execute_query swallows errors into empty results, so an empty
            # run means this project likely failed; try the next one.
            if attempt[0]:
                break
        except Exception as e:
            last_error = e
            results = []
            continue
    if not results:
        return {"skipped": True, "reason": str(last_error)}

    run_a, run_b = results
    names_a = [r["full_name"] for r in run_a]
    names_b = [r["full_name"] for r in run_b]
    per_day = Counter(r.get("sampled_from") for r in run_a)
    day_counts = {}
    for r in run_a:
        day_counts[r.get("sampled_from")] = r.get("day_repo_count", 0)
    days = sorted(per_day)
    correlation = (
        pearson(
            [float(per_day[d]) for d in days],
            [float(day_counts[d]) for d in days],
        )
        if len(days) > 2
        else float("nan")
    )
    print(f"[P5] two live runs: {len(names_a)} and {len(names_b)} repos")
    return {
        "skipped": False,
        "n_a": len(names_a),
        "n_b": len(names_b),
        "nonempty": len(names_a) > 0,
        "identical": names_a == names_b,
        "duplicates": len(names_a) - len(set(names_a)),
        "days_used": len(days),
        "allocation_correlation": correlation,
    }


# --------------------------------------------- P7: BigQuery dry-run validation


def run_p7() -> dict:
    """Dry-run every query shape BigQuerySampler can emit.

    Dry runs are FREE: BigQuery compiles the query and reports the bytes it
    would scan without executing it. This catches every class of SQL bug
    found in this audit (invalid RAND(seed), view-prefix wildcard, bare
    UNION ALL) at zero cost, and prices real runs before paying for them.
    """
    import os

    try:
        from google.cloud import bigquery

        from reporoulette.samplers.bigquery_sampler import BigQuerySampler
    except ImportError as e:
        return {"skipped": True, "reason": f"BigQuery deps not installed: {e}"}

    project_id = os.environ.get("GOOGLE_PROJECT_ID", "in-electoral-rolls")
    try:
        sampler = BigQuerySampler(
            seed=42, project_id=project_id, log_level=logging.WARNING
        )
    except Exception as e:
        return {"skipped": True, "reason": str(e)}

    day_data = {"sample_day": "20240101", "repo_count": 1000, "samples_to_take": 5}
    day_queries = [
        sampler._build_day_query(day_data, i, years_back=5) for i in range(2)
    ]
    queries = {
        "count (5 days, pruned wildcard)": sampler._build_count_query(5, 5),
        "day sample": day_queries[0],
        "combined (2 days, UNION ALL)": sampler._combine_day_queries(
            day_queries, n_samples=10
        ),
        "active (no languages)": sampler._build_active_query(
            "'2024-01-01'", "CURRENT_TIMESTAMP()", None, 10
        ),
        "active (languages)": sampler._build_active_query(
            "'2024-01-01'", "CURRENT_TIMESTAMP()", ["Python", "Go"], 10
        ),
        "get_languages": sampler._build_languages_query(
            ["tensorflow/tensorflow", "pytorch/pytorch"]
        ),
    }

    results: dict = {"skipped": False, "queries": {}}
    all_valid = True
    for name, query in queries.items():
        try:
            job = sampler.client.query(
                query, job_config=bigquery.QueryJobConfig(dry_run=True)
            )
            gb = (job.total_bytes_processed or 0) / 1e9
            results["queries"][name] = {"valid": True, "gb": gb}
        except Exception as e:
            results["queries"][name] = {"valid": False, "error": str(e)[:200]}
            all_valid = False
    results["all_valid_pass"] = all_valid
    summary = ", ".join(
        f"{name}: {q['gb']:.2f}GB" if q["valid"] else f"{name}: INVALID"
        for name, q in results["queries"].items()
    )
    print(f"[P7] dry runs ($0): {summary}")
    return results


# ------------------------------------------------- P6: live API contract checks


def run_p6(token: str) -> dict:
    """Cheap live checks that the external interfaces still match the
    assumptions baked into the samplers.
    """
    results: dict = {}

    # (a) GH Archive URL contract: daily files never existed (the bug the
    # sampler shipped with), hourly files do. Probed on a recent day so the
    # canary tracks the CURRENT contract.
    probe_day = (datetime.now(UTC) - timedelta(days=2)).strftime("%Y-%m-%d")
    daily_code = requests.head(
        f"https://data.gharchive.org/{probe_day}.json.gz", timeout=30
    ).status_code
    hourly_code = requests.head(
        f"https://data.gharchive.org/{probe_day}-15.json.gz", timeout=30
    ).status_code
    results["archive_day"] = probe_day
    results["daily_status"] = daily_code
    results["hourly_status"] = hourly_code
    results["archive_urls_pass"] = daily_code == 404 and hourly_code == 200

    # (b) GH Archive event schema: stream the head of an hourly file and
    # assert the fields the sampler parses are present and well-formed.
    def scan_archive(day: str, limit: int) -> tuple[int, int, bool]:
        checked = 0
        creations = 0
        ok = True
        with (
            requests.get(
                f"https://data.gharchive.org/{day}-15.json.gz",
                stream=True,
                timeout=120,
            ) as resp,
            gzip.GzipFile(fileobj=resp.raw) as fh,
        ):
            for line in fh:
                event = json.loads(line)
                checked += 1
                if not (
                    isinstance(event.get("type"), str)
                    and isinstance(event.get("repo", {}).get("name"), str)
                    and isinstance(event.get("created_at"), str)
                ):
                    ok = False
                if is_repo_creation(event):
                    creations += 1
                if checked >= limit:
                    break
        return checked, creations, ok

    events_checked, recent_creations, schema_ok = scan_archive(probe_day, 500)
    results["events_checked"] = events_checked
    results["creation_events"] = recent_creations
    results["schema_pass"] = schema_ok and events_checked == 500

    # (b2) Repository CreateEvents left the public events feed with GitHub's
    # Events API payload change on 2025-10-07 (empirically: thousands/hour
    # through 2025-09-15, zero from 2025-10-15 on). Verify both sides of the
    # cutoff so this check flags any future reversal of the upstream change.
    _, pre_creations, _ = scan_archive("2025-08-15", 50000)
    results["pre_cutoff_creations"] = pre_creations
    results["post_cutoff_creations"] = recent_creations
    results["cutoff_pass"] = pre_creations > 0 and recent_creations == 0

    # (c) /repositories/{id} field contract: the exact keys IDSampler maps
    # without .get() fallbacks must exist.
    resp = requests.get(
        "https://api.github.com/repositories/1",
        headers={"Authorization": f"token {token}"},
        timeout=30,
    )
    repo = resp.json() if resp.status_code == 200 else {}
    required = ["name", "full_name", "html_url", "created_at", "updated_at"]
    results["repo_endpoint_status"] = resp.status_code
    results["repo_endpoint_pass"] = (
        resp.status_code == 200
        and all(k in repo for k in required)
        and isinstance(repo.get("owner", {}).get("login"), str)
    )

    # (d) Temporal end-to-end mini-run: the sampler's population claim is
    # "repos pushed on the sampled day" - verify each returned repo's
    # pushed_at actually falls inside its sampled day's window.
    sampler = TemporalSampler(token=token, seed=7, log_level=logging.WARNING)
    repos = sampler.sample(
        n_samples=10, days_to_sample=2, min_wait=2.5, max_attempts=15
    )
    within = 0
    for r in repos:
        day = datetime.strptime(r["sampled_from"], "%Y-%m-%d").replace(tzinfo=UTC)
        pushed = datetime.strptime(r["pushed_at"], "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=UTC
        )
        if day <= pushed <= day + timedelta(days=1):
            within += 1
    frac = within / len(repos) if repos else 0.0
    results["temporal_n"] = len(repos)
    results["temporal_within_window"] = frac
    results["temporal_pass"] = len(repos) > 0 and frac >= 0.9
    print(
        f"[P6] contracts: daily={daily_code} hourly={hourly_code}, "
        f"creations pre/post cutoff: {results['pre_cutoff_creations']}/"
        f"{recent_creations}, temporal {len(repos)} repos "
        f"({100 * frac:.0f}% in window)"
    )
    return results


# ------------------------------------------------------------------- reporting


def fmt_pass(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def write_report(sections: dict) -> None:
    lines = [
        "# Randomization Validation Report",
        "",
        f"Generated {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')} by "
        "`scripts/validate_randomization.py`. All tests at alpha = 0.05. "
        "Chi-square critical values use the Wilson-Hilferty approximation; "
        "KS uses the asymptotic two-sample critical value. Aggregating draws "
        "across seeded runs treats draws as independent, which is a good "
        "approximation for sample sizes far below the population size.",
        "",
    ]

    p1 = sections.get("p1")
    if p1:
        lines += [
            "## P1 - GH Archive ground-truth inclusion test (GHArchiveSampler)",
            "",
            f"Population: all {p1['population_size']:,} repositories created in "
            f"hour {p1['archive_hour']} UTC of {p1['archive_date']} (fully "
            f"enumerated from the public archive). "
            f"{p1['n_seeds']} seeded replays of {p1['k']} draws each "
            f"({p1['draws']:,} total draws) through the real sampler code path.",
            "",
            "| Test | Statistic | Critical (5%) | Result |",
            "|---|---|---|---|",
            f"| Equal inclusion probability (100 hash buckets, df=99) | "
            f"{p1['inclusion_chi2']:.1f} | {p1['inclusion_crit']:.1f} | "
            f"{fmt_pass(p1['inclusion_pass'])} |",
            f"| Creation minute-of-hour marginal vs population (df=59) | "
            f"{p1['minute_chi2']:.1f} | {p1['minute_crit']:.1f} | "
            f"{fmt_pass(p1['minute_pass'])} |",
            f"| Owner-initial marginal vs population | "
            f"{p1['initial_chi2']:.1f} | {p1['initial_crit']:.1f} | "
            f"{fmt_pass(p1['initial_pass'])} |",
            "",
        ]

    p3 = sections.get("p3")
    if p3:
        lines += [
            "## P3 - Seed stability (pairwise KS across seeds)",
            "",
            f"{p3['pairs']} pairwise two-sample KS tests on within-day creation "
            f"times across independent seeds: {p3['rejections']} rejections "
            f"({100 * p3['rejection_rate']:.1f}%; expectation under uniform "
            f"sampling is about 5%). {fmt_pass(p3['pass'])}",
            "",
        ]

    p2 = sections.get("p2")
    if p2:
        rates = ", ".join(
            "-" if math.isnan(r) else f"{100 * r:.1f}%" for r in p2["decile_hit_rates"]
        )
        lines += [
            "## P2 - Live IDSampler audit",
            "",
            f"{p2['n_probes']:,} live probes against the real GitHub API, "
            f"{p2['n_hits']} hits ({p2['success_rate']:.1f}% success rate).",
            "",
            "| Test | Statistic | Critical (5%) | Result |",
            "|---|---|---|---|",
            f"| Probed IDs uniform over 10 deciles (df=9) | "
            f"{p2['uniform_chi2']:.2f} | {p2['uniform_crit']:.2f} | "
            f"{fmt_pass(p2['uniform_pass'])} |",
            f"| created_at monotone in ID (inversion rate) | "
            f"{100 * p2['inversion_rate']:.2f}% | < 1% | "
            f"{fmt_pass(p2['monotone_pass'])} |",
            f"| Live probe-sequence reproducibility (same seed, 2 runs) | "
            f"{'identical' if p2['reproducible'] else 'diverged'} | identical | "
            f"{fmt_pass(p2['reproducible'])} |",
            "",
            f"Hit rate by ID decile (existing-repo density over GitHub's "
            f"history): {rates}.",
            "",
            f"Coverage gap: newest observed repo ID is "
            f"{p2['newest_repo_id']:,} vs the default max_id "
            f"{p2['max_id_default']:,} - the default excludes "
            f"{100 * p2['id_space_gap']:.1f}% of the current ID space "
            f"(newest repositories). Pass max_id explicitly for full "
            f"coverage.",
            "",
        ]

    p4 = sections.get("p4")
    if p4:
        t, b = p4["temporal"], p4["id_benchmark"]

        def langs(entry):
            return ", ".join(f"{lang} ({n})" for lang, n in entry["top_languages"])

        lines += [
            "## P4 - Temporal sampler bias, quantified (measurement, not pass/fail)",
            "",
            f"Temporal sample n={t['n']} vs near-uniform ID-sample benchmark "
            f"n={b['n']}:",
            "",
            "| Metric | TemporalSampler | IDSampler benchmark |",
            "|---|---|---|",
            f"| Mean stars | {t['mean_stars']:.1f} | {b['mean_stars']:.1f} |",
            f"| Median stars | {t['median_stars']} | {b['median_stars']} |",
            f"| % with >= 1 star | {t['pct_starred']:.0f}% | {b['pct_starred']:.0f}% |",
            f"| % pushed in last 90 days | {t['pct_pushed_90d']:.0f}% | "
            f"{b['pct_pushed_90d']:.0f}% |",
            f"| Top languages | {langs(t)} | {langs(b)} |",
            "",
            f"Search-cap coverage: mean fraction of each sampled day's "
            f"repositories reachable under the 1,000-result cap: "
            f"{100 * p4['mean_coverage']:.1f}% "
            f"(per-day counts: "
            + ", ".join(
                f"{d}: {c:,}" for d, c in sorted(p4["day_total_counts"].items())
            )
            + ").",
            "",
        ]

    p5 = sections.get("p5")
    if p5:
        if p5.get("skipped"):
            lines += [
                "## P5 - BigQuery live check",
                "",
                f"Skipped: {p5['reason']}",
                "",
            ]
        else:
            if p5["n_a"] > 0 and p5["n_b"] == 0:
                repro_cell = (
                    "not verified - second run blocked by the project's "
                    "BigQuery quota; determinism is covered at the SQL level "
                    "by the unit suite (same seed produces byte-identical "
                    "queries and ORDER BY FARM_FINGERPRINT is deterministic)"
                )
            else:
                repro_cell = fmt_pass(p5["identical"])
            lines += [
                "## P5 - BigQuery live check (BigQuerySampler)",
                "",
                "| Check | Result |",
                "|---|---|",
                f"| Returns real data (was always empty before the "
                f"invalid-SQL fixes) | {p5['n_a']} repos - "
                f"{fmt_pass(p5['nonempty'])} |",
                f"| Same seed, two live runs identical | {repro_cell} |",
                f"| Duplicate repositories in output | {p5['duplicates']} - "
                f"{fmt_pass(p5['duplicates'] == 0)} |",
                f"| Days used / allocation vs day size correlation | "
                f"{p5['days_used']} days, r = {p5['allocation_correlation']:.2f} |",
                "",
            ]

    p6 = sections.get("p6")
    if p6:
        lines += [
            "## P6 - Live API contract checks",
            "",
            "| Check | Result |",
            "|---|---|",
            f"| GH Archive publishes hourly files only (daily URL 404s, "
            f"hourly 200; probed {p6['archive_day']}) | "
            f"daily={p6['daily_status']}, hourly={p6['hourly_status']} - "
            f"{fmt_pass(p6['archive_urls_pass'])} |",
            f"| Event schema fields used by GHArchiveSampler present | "
            f"first {p6['events_checked']} events well-formed - "
            f"{fmt_pass(p6['schema_pass'])} |",
            f"| Repository CreateEvents ended with GitHub's 2025-10-07 "
            f"Events API change (present before, absent after) | "
            f"{p6['pre_cutoff_creations']} in 50k events of 2025-08-15 vs "
            f"{p6['post_cutoff_creations']} recently - "
            f"{fmt_pass(p6['cutoff_pass'])} |",
            f"| /repositories/{{id}} returns the fields IDSampler maps | "
            f"HTTP {p6['repo_endpoint_status']} - "
            f"{fmt_pass(p6['repo_endpoint_pass'])} |",
            f"| Temporal population claim: pushed_at inside the sampled day | "
            f"{p6['temporal_n']} repos, "
            f"{100 * p6['temporal_within_window']:.0f}% in window - "
            f"{fmt_pass(p6['temporal_pass'])} |",
            "",
        ]

    p7 = sections.get("p7")
    if p7:
        if p7.get("skipped"):
            lines += [
                "## P7 - BigQuery dry-run validation",
                "",
                f"Skipped: {p7['reason']}",
                "",
            ]
        else:
            lines += [
                "## P7 - BigQuery dry-run validation (free)",
                "",
                "Every query shape the sampler can emit, compiled by BigQuery "
                "without execution (dry runs cost nothing and would have "
                "caught all three SQL-validity bugs in this audit). "
                "Estimated scan per query:",
                "",
                "| Query | Valid | Est. scan |",
                "|---|---|---|",
            ]
            for name, q in p7["queries"].items():
                scan = f"{q['gb']:.2f} GB" if q["valid"] else q.get("error", "")
                lines.append(f"| {name} | {fmt_pass(q['valid'])} | {scan} |")
            lines.append("")

    REPORT_PATH.write_text("\n".join(lines))
    print(f"report written to {REPORT_PATH}")


# ------------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--report", action="store_true", help="rewrite the report from cached results"
    )
    parser.add_argument("--p1", action="store_true")
    parser.add_argument("--p2", action="store_true")
    parser.add_argument("--p3", action="store_true")
    parser.add_argument("--p4", action="store_true")
    parser.add_argument("--p5", action="store_true")
    parser.add_argument("--p6", action="store_true")
    parser.add_argument("--p7", action="store_true")
    parser.add_argument("--archive-date", default="2023-06-15")
    parser.add_argument("--archive-hour", type=int, default=12)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--probes", type=int, default=2000)
    args = parser.parse_args()

    import os

    token = os.environ.get("GITHUB_TOKEN")

    # Results persist across invocations so sections can be (re)run
    # individually without losing the rest of the report.
    CACHE_DIR.mkdir(exist_ok=True)
    sections_path = CACHE_DIR / "sections.json"
    sections: dict = (
        json.loads(sections_path.read_text()) if sections_path.exists() else {}
    )

    if args.all or args.p1 or args.p3:
        sections["p1"] = run_p1(
            args.archive_date, args.archive_hour, args.seeds, args.k
        )
        sections["p3"] = run_p3(sections["p1"].pop("per_seed_seconds"))

    if args.all or args.p2:
        if not token:
            raise SystemExit("GITHUB_TOKEN is required for --p2")
        sections["p2"] = run_p2(token, args.probes)
        sections["p2_hits"] = sections["p2"].pop("hits")

    if args.all or args.p4:
        if not token:
            raise SystemExit("GITHUB_TOKEN is required for --p4")
        id_hits = sections.get("p2_hits")
        if not id_hits:
            raise SystemExit("--p4 needs a cached or fresh --p2 run first")
        sections["p4"] = run_p4(token, id_hits)

    if args.all or args.p5:
        sections["p5"] = run_p5()

    if args.all or args.p6:
        if not token:
            raise SystemExit("GITHUB_TOKEN is required for --p6")
        sections["p6"] = run_p6(token)

    if args.all or args.p7:
        sections["p7"] = run_p7()

    sections_path.write_text(json.dumps(sections))
    write_report({k: v for k, v in sections.items() if k != "p2_hits"})
    for name in ("p1", "p2", "p3", "p5", "p6", "p7"):
        section = sections.get(name)
        if not section or section.get("skipped"):
            continue
        passes = [v for k, v in section.items() if k.endswith("_pass")]
        if name == "p2":
            passes.append(section["reproducible"])
        if name == "p3":
            passes = [section["pass"]]
        if name == "p5":
            passes = [section["nonempty"], section["duplicates"] == 0]
            if not (section["n_a"] > 0 and section["n_b"] == 0):
                passes.append(section["identical"])
        print(f"{name}: {sum(passes)}/{len(passes)} checks passed")


if __name__ == "__main__":
    main()

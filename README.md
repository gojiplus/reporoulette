## RepoRoulette 🎲: Randomly Sample Repositories from GitHub

> Spin the wheel and see which GitHub repositories you get!

[![PyPI version](https://img.shields.io/pypi/v/reporoulette.svg)](https://pypi.org/project/reporoulette/)
[![Python](https://img.shields.io/pypi/pyversions/reporoulette.svg)](https://pypi.org/project/reporoulette/)
[![License](https://img.shields.io/pypi/l/reporoulette.svg)](https://pypi.org/project/reporoulette/)
[![Downloads](https://static.pepy.tech/badge/reporoulette)](https://pepy.tech/project/reporoulette)
[![CI](https://github.com/gojiplus/reporoulette/actions/workflows/ci.yml/badge.svg)](https://github.com/gojiplus/reporoulette/actions/workflows/ci.yml)

## 🚀 Installation

```bash
# Using pip
pip install reporoulette

# From source
git clone https://github.com/gojiplus/reporoulette.git
cd reporoulette
pip install -e .
```

## 📖 Sampling Methods

RepoRoulette provides three distinct methods for random GitHub repository sampling:

### 1. 🎯 ID-Based Sampling

Uses GitHub's sequential repository ID system to generate truly random samples by probing random IDs from the valid ID range. The downside of using the method is that the hit rate can be low (as many IDs are invalid, partly because the repo. is private or abandoned, etc.) And any filtering on repo. characteristics must wait till you have the names.

The function will continue to sample till either `max_attempts` or till `n_samples`. You can pass the `seed` for reproducibility.

```python
from reporoulette import IDSampler

# Initialize the sampler
sampler = IDSampler(token="your_github_token")

# Get 50 random repositories
repos = sampler.sample(n_samples=50)

# Print basic stats
print(f"Success rate: {sampler.success_rate:.2f}%")
print(f"Samples collected: {len(repos)}")
```

### 2. ⏱️ Temporal Sampling

Randomly selects days within a specified date range and retrieves repositories updated during those periods using weighted sampling based on repository activity.

```python
from reporoulette import TemporalSampler
from datetime import datetime, timedelta

# Define a date range (last 3 months)
end_date = datetime.now()
start_date = end_date - timedelta(days=90)

# Initialize the sampler
sampler = TemporalSampler(
    token="your_github_token",
    start_date=start_date,
    end_date=end_date
)

# Get 100 random repositories
repos = sampler.sample(n_samples=100)

# Get repositories with specific characteristics
filtered_repos = sampler.sample(
    n_samples=50,
    min_stars=10,
    language="python"  # Note: single language, not list
)
```

### 3. 🔍 BigQuery Sampling

The `BigQuerySampler` leverages Google BigQuery's public GitHub dataset to sample repositories with advanced filtering capabilities.

#### Setup for BigQuery Sampler

1. **Create a Google Cloud Platform (GCP) project**:
   - Go to the [GCP Console](https://console.cloud.google.com/)
   - Create a new project

2. **Enable the BigQuery API**:
   - In your project, go to "APIs & Services" > "Library"
   - Search for "BigQuery API" and enable it

3. **Create a service account**:
   - Go to "IAM & Admin" > "Service Accounts"
   - Create a new service account
   - Grant it the "BigQuery User" role
   - Create and download a JSON key file

4. **Install required dependencies**:
   ```bash
   pip install google-cloud-bigquery google-auth
   ```

5. **Using BigQuerySampler**:

```python
from reporoulette import BigQuerySampler

# Initialize with service account credentials
sampler = BigQuerySampler(
    credentials_path="path/to/your-service-account-key.json",
    project_id="your-gcp-project-id",
    seed=42
)

# Sample active repositories with commits in the last year
active_repos = sampler.sample(
    n_samples=50,
    population="active",
    languages=["Python", "JavaScript"]  # Optional language filter
)

# Sample repositories across random days
random_repos = sampler.sample_by_day(
    n_samples=50,
    days_to_sample=10,
    years_back=5
)

# Get language information for sampled repositories
languages = sampler.get_languages(random_repos)

# Print results
for repo in random_repos:
    print(f"Repository: {repo['full_name']}")
    repo_languages = languages.get(repo['full_name'], [])
    if repo_languages:
        print(f"Primary language: {repo_languages[0]['language']}")
    print("---")
```

**Advantages:**
- Handles large sample sizes efficiently
- Powerful filtering and stratification options
- Not limited by GitHub API rate limits
- Access to historical data

**Limitations:**
- Could be expensive
- Requires Google Cloud Platform account and billing
- Dataset may have a slight delay (typically 24-48 hours)

### 4. GH Archive Sampler

The `GHArchiveSampler` fetches repositories by sampling events from [GitHub Archive](https://www.gharchive.org/), a project that records the public GitHub timeline.

```python
from reporoulette import GHArchiveSampler

# Initialize with optional parameters
sampler = GHArchiveSampler(seed=42)  # Set seed for reproducibility

# Sample repositories
repos = sampler.sample(
    n_samples=100,           # Number of repositories to sample
    days_to_sample=5,        # Number of random days to sample from
    repos_per_day=20,        # Repositories to sample per day
    years_back=2,            # How many years to look back
    event_types=["PushEvent", "CreateEvent", "PullRequestEvent"]  # Event types to consider
)

# Access results
for repo in repos:
    print(f"Repository: {repo['full_name']}")
    print(f"Event Type: {repo['event_type']}")
    print(f"Sampled From: {repo['sampled_from']}")
    print("---")
```


## 📊 Example Use Cases

- **Academic Research**: Study coding practices across different languages and communities
- **Learning Resources**: Discover diverse code examples for education
- **Data Science**: Build datasets for machine learning models about code patterns
- **Trend Analysis**: Identify emerging technologies and practices
- **Security Research**: Find vulnerability patterns across repository types

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Projects

- [GHTorrent](https://ghtorrent.org/) - GitHub data archive project
- [GitHub Archive](https://www.githubarchive.org/) - Archive of public GitHub timeline
- [PyGithub](https://github.com/PyGithub/PyGithub) - Python library for the GitHub API

---

Built with ❤️ by Gojiplus

## Repo Roulette 🎲

> Spin the wheel and see which GitHub repositories you get!

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/reporoulette.svg)](https://pypi.org/project/reporoulette/)
[![License](https://img.shields.io/pypi/l/reporoulette.svg)](https://pypi.org/project/reporoulette/)
[![Downloads](https://static.pepy.tech/badge/reporoulette)](https://pepy.tech/project/reporoulette)

Randomly sample repositories from GitHub. 

## 🌟 Features

- **Multiple sampling methods** to meet different research needs
- **Avoid the 1,000 result limitation** of GitHub's Search API

## 🚀 Installation

```bash
# Using pip
pip install reporoulette

# From source
git clone https://github.com/username/reporoulette.git
cd reporoulette
pip install -e .
```

## 📖 Sampling Methods

RepoRoulette provides three distinct methods for random GitHub repository sampling:

### 1. 🎯 ID-Based Sampling

Uses GitHub's sequential repository ID system to generate truly random samples by probing random IDs from the valid ID range.

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

**Advantages:**
- Most unbiased sampling method
- Covers the entire GitHub ecosystem
- Simple approach with minimal API calls

**Limitations:**
- Lower hit rate (many IDs are invalid)
- No control over repository characteristics

### 2. ⏱️ Temporal Sampling

Randomly selects time points (date/hour combinations) within a specified range and then retrieves repositories updated during those periods.

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
    languages=["python", "javascript"]
)
```

**Advantages:**
- Higher hit rate than ID-based sampling
- Can filter by repository characteristics
- Allows for stratified sampling by time periods

**Limitations:**
- Limited to repositories with recent activity
- Some time periods may have fewer repositories

### 3. 🔍 BigQuery Sampling

Leverages Google BigQuery's GitHub dataset for high-volume, efficient sampling. Perfect for research requiring large samples or specific criteria.

```python
from reporoulette import BigQuerySampler

# Initialize the sampler (requires GCP credentials)
sampler = BigQuerySampler(
    credentials_path="path/to/credentials.json"
)

# Sample 1,000 repositories created in the last year
repos = sampler.sample(
    n_samples=1000,
    created_after="2023-01-01",
    sample_by="created_at"
)

# Sample repositories with multiple criteria
specialty_repos = sampler.sample(
    n_samples=500,
    min_stars=100,
    min_forks=50,
    languages=["rust", "go"],
    has_license=True
)
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
- Requires knowledge of SQL for custom queries

## 📊 Example Use Cases

- **Academic Research**: Study coding practices across different languages and communities
- **Learning Resources**: Discover diverse code examples for education
- **Data Science**: Build datasets for machine learning models about code patterns
- **Trend Analysis**: Identify emerging technologies and practices
- **Security Research**: Find vulnerability patterns across repository types

## 🛠️ Configuration

Create a `.reporoulette.yml` file in your home directory or project root:

```yaml
github:
  token: "your_github_token"
  rate_limit_safety: 100  # Stop when this many requests remain

bigquery:
  credentials_path: "path/to/credentials.json"
  project_id: "your-gcp-project"

sampling:
  default_method: "temporal"
  cache_results: true
  cache_path: "~/.reporoulette/cache"
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Related Projects

- [GHTorrent](https://ghtorrent.org/) - GitHub data archive project
- [GitHub Archive](https://www.githubarchive.org/) - Archive of public GitHub timeline
- [PyGithub](https://github.com/PyGithub/PyGithub) - Python library for the GitHub API

---

Built with ❤️ by [Your Name/Organization]

# gin-bids-py-analysis

BIDS-based iEEG analysis pipeline.

## Installation

```bash
git clone <repo-url>
cd gin-bids-py-analysis
pip install -e ".[dev]"
```

## Project Structure

```
src/gin_bids_py_analysis/
├── bids/         BIDS file discovery & querying (wraps PyBIDS)
├── data/         iEEG data loaders (placeholder — see data/__init__.py)
└── processing/   Analysis pipeline (compute and I/O separated)
    └── hilbert/  Hilbert-transform analysis (stub)
    └── trial_stats/ Subject-level trial statistics on Hilbert derivatives
scripts/          Runnable analysis scripts (argparse + joblib)
tests/            pytest test suite
```

## Quick Start

```python
from gin_bids_py_analysis.bids import BIDSDataset

# derivatives=True by default: source and derivative files are queryable together
ds = BIDSDataset("/path/to/bids")
files = ds.get_files(subject="01", suffix="ieeg", extension=".vhdr")
subjects = ds.get_subjects()

# Any BIDS entity is queryable
print(files[0]["acq"])       # dynamic entity access
print(files[0].entities)     # full entity dict
```

## Running Tests

```bash
pytest
pytest --cov=gin_bids_py_analysis --cov-report=term-missing
```

## Architecture

See [.github/copilot-instructions.md](.github/copilot-instructions.md) for the full architecture documentation, design rules, and patterns for adding new analyses.

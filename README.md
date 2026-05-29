# bidsforge

**BIDS-based iEEG analysis pipelines for Hilbert envelopes, subject-level trial statistics, and group-level ROI statistics.**

`bidsforge` runs analysis workflows on BIDS datasets and writes BIDS derivatives under `derivatives/<pipeline_label>/`. It is designed for projects where iEEG recordings, events, behavior tables, and electrode localizations are already organized in BIDS or BIDS-like form and need to be processed consistently.

## When Should I Use `bidsforge`?

Use `bidsforge` when you need to:

- extract Hilbert-band envelopes from BIDS iEEG recordings;
- compare two trial conditions at subject level;
- regress trial activity against a continuous predictor at subject level;
- aggregate subject-level statistics into group-level ROI results;
- keep processing parameters and derivative provenance close to the generated outputs.

The normal workflow is:

```text
BIDS dataset
  -> select files with BIDSDataset / BIDS entities
  -> configure pipeline params
  -> configure writer params
  -> run processor.run(...)
  -> BIDS derivatives + dataset_description.json
```

## Requirements

- Python `>=3.10`
- A BIDS dataset readable by `pybids`
- A project-local virtual environment is recommended

Core dependencies are declared in [pyproject.toml](pyproject.toml): `pybids`, `pydantic`, `mne`, `numpy`, `pandas`, `scipy`, `h5py`, `pybv`, `pyfftw`, `joblib`, and `tqdm`.

## Installation

Install in editable mode:

```bash
git clone <repo-url>
cd bidsforge

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

Optional extras:

```bash
python -m pip install -e ".[viz]"   # visualization UI dependencies
python -m pip install -e ".[dev]"   # tests + visualization tooling
```

## Very Quick Usage

Minimal Hilbert run from raw BrainVision iEEG files:

```python
from bidsforge.bids import BIDSDataset
from bidsforge.processing.hilbert import (
    HilbertParams,
    HilbertProcessing,
    HilbertProcessingWriter,
    HilbertWriterParams,
)

ds = BIDSDataset("D:/data/bids")
files = ds.get_files(scope="raw", suffix="ieeg", extension=".vhdr")

processor = HilbertProcessing(HilbertParams(f_min=50, f_max=150, f_step=10))
writer = HilbertProcessingWriter(
    HilbertWriterParams(bids_root=ds.root, output_format="hdf5")
)

out_paths = processor.run(files, writer, n_jobs=1, skip_existing=True)
```

Repository-specific scripts can live under `scripts/` when they make repeated analyses easier. They are intentionally editable recipes, not a stable command-line interface.

## Built-In Public Pipelines

- `hilbert`: extracts Hilbert-band envelopes from iEEG recordings.
- `condition_test`: compares two trial conditions at subject level.
- `regression`: computes condition-specific trial-wise regression at subject level.
- `condition_test_group`: aggregates `condition_test` outputs into group-level ROI statistics.
- `regression_group`: aggregates `regression` outputs into group-level ROI statistics.

Typical order:

```text
raw iEEG
  -> hilbert
  -> condition_test or regression
  -> condition_test_group or regression_group
```

## Documentation

Start here:

- [docs/README.md](docs/README.md): documentation map
- [docs/end_to_end_guide.md](docs/end_to_end_guide.md): first complete analysis workflow
- [docs/troubleshooting_guide.md](docs/troubleshooting_guide.md): common setup, BIDS discovery, processing, and output issues

Pipeline guides:

- [docs/pipelines/hilbert.md](docs/pipelines/hilbert.md)
- [docs/pipelines/condition_test.md](docs/pipelines/condition_test.md)
- [docs/pipelines/regression.md](docs/pipelines/regression.md)
- [docs/pipelines/condition_test_group.md](docs/pipelines/condition_test_group.md)
- [docs/pipelines/regression_group.md](docs/pipelines/regression_group.md)

Developer guide:

- [docs/add_pipeline_guide.md](docs/add_pipeline_guide.md): add a new processing pipeline

## Visualization

Visualization helpers live under `bidsforge.visualization.trial_stats`.

Install visualization dependencies first:

```bash
python -m pip install -e ".[viz]"
```

## Development

If you want to contribute to project development, please contact Benjamin BONTEMPS first. The repository you can access is most likely the public repository, not the active development repository.

Run tests:

```bash
pytest -v
pytest --cov=bidsforge --cov-report=term-missing
```

## License

**bidsforge** is released under the **BSD-3-Clause License**.

Copyright © 2026  
Benjamin BONTEMPS

This license allows unrestricted use, modification, and redistribution of the software, while prohibiting the use of the author's name for endorsement without permission. See the [`LICENSE`](LICENSE) file for details.

## Acknowledgements

This project has received funding from the European Union's Research and Innovation Program Horizon Europe under Grant Agreement [No. 101147319](https://doi.org/10.3030/101147319) (EBRAINS 2.0).

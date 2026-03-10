# GitHub Copilot Instructions — gin-bids-py-analysis

> **This is the live architecture document.**
> Update it whenever a module's responsibilities, public API, or design rules change.

---

## Project Overview

`gin-bids-py-analysis` is a Python package for BIDS-based iEEG analysis.
It wraps PyBIDS for file discovery, provides a typed query interface over an
entire dataset (source + derivatives in one object), and implements a strict
separation between **computation** and **file I/O** in the processing pipeline.

| Item | Value |
|---|---|
| **Pip name** | `gin-bids-py-analysis` |
| **Import name** | `gin_bids_py_analysis` |
| **Python** | ≥ 3.10 |
| **Dep management** | plain `pip` + `setuptools`; install with `pip install -e ".[dev]"` |
| **Test runner** | `pytest` |

---

## Module Map

```
src/gin_bids_py_analysis/
├── bids/           BIDS file discovery & querying (wraps PyBIDS)
├── data/           iEEG data loaders — PLACEHOLDER, see data/__init__.py
└── processing/     Analysis pipeline (compute and I/O are separated)
    └── hilbert/    Hilbert-transform analysis subpackage (stub)
scripts/            Runnable argparse scripts; use joblib for parallelism
tests/              pytest test suite (mirrors src/ structure)
```

---

## Key Classes

### `bids/`

| Class | File | Responsibility |
|---|---|---|
| `BIDSDataset` | `dataset.py` | Primary entry-point. Wraps `pybids.BIDSLayout`. **Derivatives are included by default** (`derivatives=True`). |
| `BIDSSubject` | `subject.py` | One participant; holds `list[BIDSFile]`; filterable via `.get_files(**entities)`. |
| `BIDSFile` | `file.py` | Wraps a pybids `BIDSFile`. Exposes `.entities: dict`, `file['any_entity']`, `.suffix`, `.extension`, `.path`. |
| `BIDSFileGroup` | `file_group.py` | Groups a `primary: BIDSFile` with optional `secondaries: list[BIDSFile]`. The unit passed to processors. |
| helpers | `helpers.py` | `build_bids_path(entities, root, suffix, extension)` → `Path`; `parse_entities(path)` → `dict`. |

### `processing/`

| Class | File | Responsibility |
|---|---|---|
| `BaseProcessing` | `base.py` | Abstract. Concrete `execute(groups, n_jobs) -> list[BaseProcessingResult]` fans out to abstract `process_group(group) -> BaseProcessingResult`. **No file I/O allowed here.** |
| `BaseProcessingResult` | `base.py` | Abstract dataclass. `source_group: BIDSFileGroup`, `metadata: dict`. Subclassed per analysis. |
| `BaseProcessingWriter` | `base.py` | Abstract. `write(result, output_root) -> Path`. Auto-builds BIDS output path from `result.source_group.primary`'s entities + pipeline label. |

---

## Design Rules

1. **No I/O in processors.**
   `BaseProcessing.process_file()` is pure computation. All disk writes go through a
   `BaseProcessingWriter` subclass. This keeps processors independently testable
   and safe for parallel execution.

2. **`execute()` is not overridable.**
   `BaseProcessing.execute(groups, n_jobs)` is fully implemented in the base class:
   it fans out to `process_group()` using joblib. Subclasses only override `process_group()`.

2. **All BIDS entities are queryable.**
   `BIDSFile` exposes `.entities: dict` and `__getitem__`.
   Never hard-code only the "standard" entities (`subject`, `session`, `task`).

3. **No `BIDSDerivativesDataset` class.**
   `BIDSDataset(root, derivatives=True)` (the default) includes derivative files.
   There is no separate derivatives dataset class.

4. **Per-analysis subclasses.**
   Each analysis in `processing/<name>/` must define:
   - `<Name>Params(BaseModel)` — Pydantic v2
   - `<Name>ProcessingResult(BaseProcessingResult)` — dataclass
   - `<Name>ProcessingWriter(BaseProcessingWriter)` — uses `build_bids_path()` from `result.source_group.primary`
   - `<Name>Processing(BaseProcessing)` — overrides `process_group(group)` only; receives params via constructor

5. **Pydantic v2 for all processing params.**
   Pass the params object to the processor constructor, not to `execute()`.

6. **Parallelism at the script level.**
   Use `joblib` in `scripts/` for multi-subject parallelism.
   Processors must be stateless and side-effect-free (no shared mutable state).

7. **`data/` is a placeholder.**
   Do not implement data loading until the source project module is available.
   Refer to `data/__init__.py` for the expected interface stub.

---

## Adding a New Analysis

1. Create `src/gin_bids_py_analysis/processing/<name>/` with:
   ```
   __init__.py      — re-export the four classes below
   params.py        — <Name>Params(BaseModel)
   result.py        — <Name>ProcessingResult(BaseProcessingResult)
   writer.py        — <Name>ProcessingWriter(BaseProcessingWriter)
   processor.py     — <Name>Processing(BaseProcessing)
   ```
2. In `writer.py`, use `build_bids_path()` from `gin_bids_py_analysis.bids.helpers`
   to construct the output path automatically from `source_file.entities`.
3. Add tests in `tests/processing/<name>/`.
4. **Update this file**: add the new analysis to the module map and key classes tables.

---

## Typical Usage

### Querying the dataset

```python
from gin_bids_py_analysis.bids import BIDSDataset

ds = BIDSDataset("/path/to/bids")          # derivatives=True by default
files = ds.get_files(subject="01", suffix="ieeg", extension=".vhdr")
subjects = ds.get_subjects()

# Any BIDS entity is queryable
print(files[0]["acq"])           # subscript — raises KeyError if missing
print(files[0].get("run"))       # safe get — returns None if missing
print(files[0].entities)         # full entity dict
```

### Running a processing step

```python
from pathlib import Path
from gin_bids_py_analysis.bids import BIDSDataset, BIDSFileGroup
from gin_bids_py_analysis.processing.hilbert import (
    HilbertParams, HilbertProcessing, HilbertProcessingWriter,
)

ds = BIDSDataset("/path/to/bids")
files = ds.get_files(subject="01", suffix="ieeg")

params = HilbertParams(freq_bands=[(1, 4), (8, 12)], sfreq=1000.0)
processor = HilbertProcessing(params)
writer = HilbertProcessingWriter(Path("/path/to/bids"))

# Simplest case — pass files directly (auto-wrapped into single-file groups)
out_paths = processor.run(files, writer)
out_paths = processor.run(files, writer, n_jobs=4)

# Multi-modal groups — build BIDSFileGroup explicitly when you need secondaries
physio_files = ds.get_files(subject="01", suffix="physio")
groups = [
    BIDSFileGroup(primary=ieeg, secondaries=[physio])
    for ieeg, physio in zip(files, physio_files)
]
out_paths = processor.run(groups, writer)

# collect all results in memory (useful for testing / interactive use)
results = processor.execute(files)              # sequential
results = processor.execute(files, n_jobs=4)    # parallel
```

### Multi-subject parallel run (in a script)

Use `run()` — it processes and writes each file immediately, keeping only one result in memory at a time:

```python
files = ds.get_files(suffix="ieeg")
out_paths = processor.run(files, writer, n_jobs=-1)  # all CPUs
```

---

## Running Tests

```bash
pip install -e ".[dev]"
pytest
pytest --cov=gin_bids_py_analysis --cov-report=term-missing
```

Tests that require a real pybids-indexed dataset are marked `@pytest.mark.skip`
and must be enabled explicitly once pybids installation is verified.

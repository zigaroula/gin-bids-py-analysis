# GitHub Copilot Instructions - gin-bids-py-analysis

> This is the live architecture document.
> Update it whenever a module's responsibilities, public API, or design rules change.

---

## Project Overview

`gin-bids-py-analysis` is a Python package for BIDS-based iEEG analysis.
It wraps PyBIDS for file discovery, provides a typed query interface over an
entire dataset (source + derivatives in one object), and enforces a strict
separation between numerical computation and file I/O in the processing layer.

| Item | Value |
|---|---|
| Pip name | `gin-bids-py-analysis` |
| Import name | `gin_bids_py_analysis` |
| Python | >= 3.10 |
| Dependency management | plain `pip` + `setuptools`; install with `pip install -e ".[dev]"` |
| Virtual env | `.venv` at the project root - prefer `.venv\Scripts\pip`, `.venv\Scripts\python`, and `.venv\Scripts\pytest` |
| Test runner | `pytest` |

---

## Module Map

```text
src/gin_bids_py_analysis/
|- bids/                BIDS file discovery, wrapping, and querying
|- data/
|  `- loader.py         iEEG loading helpers; `load_ieeg()` wraps `mne.io.read_raw`
`- processing/
   |- base.py           shared processor / writer framework
   |- utils/
   |  `- channels.py    shared mono / bipolar montage enums and helpers
   |- hilbert/          reference end-to-end processing subpackage
   |  |- __init__.py    public re-exports
   |  |- dsp.py         pure numerical pipeline, no file I/O
   |  |- fir.py         reusable FIR / Hilbert DSP primitives
   |  |- params.py      `HilbertParams` + `HilbertWriterParams`
   |  |- processor.py   BIDS/MNE orchestration only
   |  |- result.py      `HilbertProcessingResult`
   |  `- writer.py      HDF5 + BrainVision output
   `- delphos/          detector code not yet refactored to the full processing pattern
scripts/                edit-and-run scripts for local workflows
tests/                  pytest suite mirroring `src/`
```

---

## Key Classes and Helpers

### `bids/`

| Class / helper | File | Responsibility |
|---|---|---|
| `BIDSDataset` | `dataset.py` | Primary entry-point. Wraps `pybids.BIDSLayout`. Derivatives are included by default (`derivatives=True`). |
| `BIDSSubject` | `subject.py` | One participant; holds `list[BIDSFile]`; filterable via `.get_files(**entities)`. |
| `BIDSFile` | `file.py` | Wraps a PyBIDS file. Exposes `.entities`, subscript access, `.suffix`, `.extension`, and `.path`. |
| `BIDSFileGroup` | `file_group.py` | Groups a `primary: BIDSFile` with optional `secondaries: list[BIDSFile]`. This is the unit passed to processors. |
| `build_bids_path` / `parse_entities` | `helpers.py` | Build derivative paths and recover entities from paths. |

### `data/`

| Function | File | Responsibility |
|---|---|---|
| `load_ieeg` | `loader.py` | Loads iEEG recordings from a `BIDSFile` via `mne.io.read_raw`, using extension-based autodetection. |

### `processing/`

| Class / helper | File | Responsibility |
|---|---|---|
| `BaseProcessing` | `base.py` | Abstract processor. Subclasses implement `process_group(group, progress_tracking_position=0)` only. `execute()` and `run()` are concrete and handle joblib parallelism. |
| `BaseProcessingParams` | `base.py` | Pydantic v2 base for algorithm parameters. |
| `BaseProcessingResult` | `base.py` | Abstract dataclass with `source_group` and `metadata`. Subclassed per analysis. |
| `BaseProcessingWriter` | `base.py` | Abstract writer. `write(result)` builds the BIDS derivative path and delegates serialization to `_write_data(result, output_path)`. |
| `BaseWriterParams` | `base.py` | Pydantic v2 base for writer configuration. Includes `bids_root`, `pipeline_label`, `output_modality`, `output_description`, `output_suffix`. Subclasses declare `output_format` (a `Literal` type) and override `output_extension` as a `@computed_field`. |
| `MontageMode`, `BipolarDirection`, `BipolarStorage`, `build_montage` | `processing/utils/channels.py` | Shared channel re-referencing utilities for analyses that work on raw or derived channel montages. |

### `processing/hilbert/` as the reference implementation

Use `processing/hilbert/` as the model for new processings.

| Module | Responsibility |
|---|---|
| `params.py` | Keeps both algorithm params and writer params in one place. |
| `dsp.py` | Holds the main pure array-based pipeline for Hilbert. |
| `fir.py` | Holds lower-level pure DSP primitives reused by the Hilbert pipeline. |
| `processor.py` | Loads data, subsets channels, forwards pure arrays into pure-computation helpers, and assembles a result object. |
| `result.py` | Carries processed arrays plus the metadata and source information writers need. |
| `writer.py` | Dispatches on `output_format` (via the computed `output_extension`) and serializes one result into one or more derivative files. |

---

## Design Rules

1. No file I/O in processors.
   `BaseProcessing.process_group()` may load source data and build in-memory results, but all output writes go through a `BaseProcessingWriter` subclass.

2. Keep orchestration separate from numerical code.
   Most heavy numpy/scipy logic should live in one or more dedicated pure-computation modules so it can be tested with synthetic arrays. `processor.py` should stay focused on dataset objects, loading, metadata, and result assembly. `dsp.py` is a common name for one of those modules, but it is not required.

3. `process_group()` owns one `BIDSFileGroup`.
   Its signature is `process_group(group, progress_tracking_position=0)`. Analyses should accept both bare `BIDSFile` objects and pre-built `BIDSFileGroup`s through the inherited `execute()` / `run()` methods.

4. Do not override `execute()` or `run()`.
   These are implemented in `BaseProcessing`. `run()` also writes `dataset_description.json` once before processing and includes `processor.params` under `GeneratedBy[0]["Parameters"]` when available.

5. Per-analysis package layout is standardized.
   New processing subpackages should usually contain:
   - `__init__.py` for public re-exports
   - `params.py` for both `<Name>Params` and `<Name>WriterParams`
   - one or more pure-computation modules when the analysis has substantial numerical logic (for example `dsp.py`, `fir.py`, or other focused helpers)
   - `processor.py` for orchestration
   - `result.py` for the result dataclass
   - `writer.py` for serialization

6. Use Pydantic v2 for params and dataclasses for results.
   Algorithm and writer config belong in Pydantic models. In-memory outputs belong in dataclasses derived from `BaseProcessingResult`.

7. Reuse shared montage helpers for channel re-referencing.
   If an analysis supports mono / bipolar processing, use `processing.utils.channels` rather than reimplementing channel parsing or adjacency rules. If channels must be subset before montage, subset both the data matrix and channel names together and preserve original file order.

8. Result objects must carry writer-relevant metadata.
   Follow the Hilbert pattern: include processed arrays, channel labels, sampling frequencies, algorithm metadata, and any source annotations or events needed for later export. Writers should not have to reload the raw recording to recover metadata they can receive in the result.

9. Writers may support multiple output formats.
   Each concrete `BaseWriterParams` subclass declares an `output_format` field (a `Literal` type enumerating valid format names) and overrides `output_extension` as a `@computed_field` that maps those names to file extensions. `_write_data()` may fan out into multiple files derived from the same `output_path`, but path construction itself remains the base writer's job.

10. Prefer structured derivative outputs.
   When writing HDF5-like outputs, store data plus explicit axes, metadata, and provenance rather than raw arrays alone. Deterministic axis ordering and sorted keys are preferred.

11. Keep optional dependencies isolated.
   If a writer backend depends on an optional package, prefer lazy imports or explicit runtime errors at write time so importing computation modules does not require every output dependency.

12. Docstrings are part of the contract.
   Keep module, class, and function docstrings aligned with actual behavior, especially for shapes, dtypes, units, optional steps, naming conventions, and file schemas. If behavior changes, update docstrings and tests in the same change.

13. All BIDS entities are queryable.
   `BIDSFile` exposes `.entities` and `__getitem__`. Do not assume only `subject`, `session`, and `task` matter.

14. There is no separate derivatives dataset type.
   `BIDSDataset(root, derivatives=True)` already includes derivative files.

---

## Reusable Patterns from Hilbert

### Processor pattern

- Load source signals with `load_ieeg()` in `processor.py`.
- Convert data to the working dtype early if needed (`float32` in Hilbert).
- Keep channel-name transformations explicit and synchronized with the data rows.
- Pass plain arrays and plain metadata into pure-computation helpers.
- Return one rich result object per input group.

### Pure computation pattern

- Put pure computation in standalone functions that accept numpy arrays and params.
- Split that computation across as many focused files as needed.
- `dsp.py` is a common top-level name, but the important rule is separation of concerns, not the filename.
- In Hilbert, the pure computation is split between `dsp.py` and `fir.py`.
- Precompute invariants once per file when they are reused across channels.
- Keep pure-computation helpers free of file-system side effects so they are safe for unit tests and joblib workers.

### Writer pattern

- Let `BaseProcessingWriter.write()` construct the canonical output path.
- Store provenance such as source path, pipeline name, and package version.
- If source annotations are preserved in the result, remap them to the output sample rate during writing rather than recomputing them from disk.
- If one result expands to multiple derivative files, derive those filenames from the base `output_path` rather than rebuilding BIDS paths from scratch.

### Testing pattern

- Add numpy-only tests for pure DSP and helper functions.
- Add processor tests that mock `load_ieeg()` and verify shape, dtype, metadata, and channel-selection behavior.
- Add writer tests that verify BIDS paths, file structure, and format-specific regressions.
- Keep tests focused on contract-level behavior rather than implementation details.

---

## Adding a New Analysis

1. Create `src/gin_bids_py_analysis/processing/<name>/`.
2. Add `__init__.py` that re-exports the public classes for the analysis.
3. Add `params.py` with:
   - `<Name>Params(BaseProcessingParams)`
   - `<Name>WriterParams(BaseWriterParams)`
4. Add one or more pure-computation modules if the analysis has non-trivial numerical logic.
5. Add `result.py` with `<Name>ProcessingResult(BaseProcessingResult)`.
6. Add `writer.py` with `<Name>ProcessingWriter(BaseProcessingWriter)`.
7. Add `processor.py` with `<Name>Processing(BaseProcessing)`.
8. If the analysis operates on iEEG recordings, prefer `load_ieeg()` rather than format-specific readers in the processor.
9. Add tests under `tests/processing/<name>/`.
10. Update this file whenever the new analysis changes the reusable architecture.

---

## Typical Usage

### Querying the dataset

```python
from gin_bids_py_analysis.bids import BIDSDataset

ds = BIDSDataset("/path/to/bids")  # derivatives=True by default
files = ds.get_files(subject="01", suffix="ieeg", extension=".vhdr")
subjects = ds.get_subjects()

print(files[0]["acq"])      # raises KeyError if missing
print(files[0].get("run"))  # safe get
print(files[0].entities)    # full entity dict
```

### Running the Hilbert processing

```python
from pathlib import Path

from gin_bids_py_analysis.bids import BIDSDataset, BIDSFileGroup
from gin_bids_py_analysis.processing.hilbert import (
    HilbertParams,
    HilbertProcessing,
    HilbertProcessingWriter,
    HilbertWriterParams,
    MontageMode,
)

ds = BIDSDataset("/path/to/bids")
files = ds.get_files(subject="01", suffix="ieeg")

params = HilbertParams(
    f_min=50,
    f_max=150,
    f_step=10,
    downsampled_frequency_hz=64.0,
    montage_mode=MontageMode.MONO,
)
processor = HilbertProcessing(params)
writer = HilbertProcessingWriter(
    HilbertWriterParams(bids_root=Path("/path/to/bids"))
)

out_paths = processor.run(files, writer)
out_paths = processor.run(files, writer, n_jobs=4)

results = processor.execute(files)
results = processor.execute(files, n_jobs=4)

physio_files = ds.get_files(subject="01", suffix="physio")
groups = [
    BIDSFileGroup(primary=ieeg, secondaries=[physio])
    for ieeg, physio in zip(files, physio_files)
]
out_paths = processor.run(groups, writer)
```

### Switching writer format

```python
writer = HilbertProcessingWriter(
    HilbertWriterParams(
        bids_root=Path("/path/to/bids"),
        output_format="brainvision",
    )
)
```

---

## Running Tests

```bash
.venv\Scripts\pip install -e ".[dev]"
.venv\Scripts\pytest
.venv\Scripts\pytest tests/processing/hilbert -q
.venv\Scripts\pytest --cov=gin_bids_py_analysis --cov-report=term-missing
```

Tests that require a real PyBIDS-indexed dataset may be skipped until the local
environment is fully configured.

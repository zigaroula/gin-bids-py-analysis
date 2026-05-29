# Troubleshooting Guide

This guide helps diagnose the most common `bidsforge` setup, BIDS discovery, processing, and writer issues.

## Quick Diagnosis First

Run these checks before deep debugging:

```bash
python -c "import sys; print(sys.executable)"
python -m pip show bidsforge
python -c "import bidsforge; print(bidsforge.__file__)"
pytest -q
```

For dataset discovery:

```python
from bidsforge.bids import BIDSDataset

ds = BIDSDataset("D:/data/bids")
print(len(ds.get_files(scope="raw", suffix="ieeg")))
print(len(ds.get_files(scope="hilbert", suffix="ieeg", extension=".h5")))
```

## Installation and Environment

### Issue: `ModuleNotFoundError: No module named 'bidsforge'`

**Problem**: the package is not installed in the Python environment used by your shell or IDE.

**Solution**:

```bash
cd bidsforge
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -c "import bidsforge; print(bidsforge.__file__)"
```

### Issue: tests or scripts use the wrong Python

**Problem**: `python`, `pip`, and your IDE may point to different interpreters.

**Solution**:

- Prefer `python -m pip ...` over bare `pip`.
- Print `sys.executable` from the same terminal that runs scripts.
- In the IDE, select the repository `.venv` interpreter.

## BIDS Discovery

### Issue: no files are found

**Problem**: filters do not match BIDS entities or derivatives were not indexed.

**Solution**:

1. Start broad, then narrow:

```python
ds.get_files()
ds.get_files(suffix="ieeg")
ds.get_files(scope="raw", suffix="ieeg")
ds.get_files(scope="raw", suffix="ieeg", extension=".vhdr")
```

2. Check exact extensions, including the leading dot: `.vhdr`, `.h5`, `.mat`, `.tsv`.
3. Check `desc` values. `desc-bandenv_ieeg.h5` is queried with `desc="bandenv"`.
4. Recreate the `BIDSDataset` object after writing new derivatives so PyBIDS re-indexes them.

### Issue: derivative files are not visible

**Problem**: derivatives need a valid `dataset_description.json`.

**Solution**:

- Run pipelines through `processor.run(...)`; it writes derivative `dataset_description.json`.
- If files were copied manually, add a BIDS derivative description under `derivatives/<pipeline>/dataset_description.json`.
- Re-instantiate `BIDSDataset(root)` after writing derivatives.

## Subject Grouping

### Issue: subject-level pipelines cannot find behavior or electrodes tables

**Problem**: secondary filters do not match the table entities.

**Solution**:

Use explicit filters:

```python
secondary_filters = [
    {"scope": "raw", "datatype": "beh", "suffix": "beh", "extension": ".tsv"},
    {"scope": "raw", "datatype": "ieeg", "suffix": "electrodes", "extension": ".tsv"},
    {"scope": "raw", "datatype": "ieeg", "suffix": "events", "extension": ".tsv"},
]
```

If a table is task-specific, include `task`. If it is subject-level and has no task, leave `task` out.

### Issue: ambiguous matching errors

**Problem**: multiple candidate files match the same subject/run with the same specificity.

**Solution**:

- Add distinguishing BIDS entities to filenames or filters, such as `task`, `run`, `ses`, or `desc`.
- Remove duplicate generic sidecars.
- Prefer one clear table per subject/run for each resolver role.

## Hilbert

### Issue: invalid frequency range

**Problem**: `f_max <= f_min` or `f_step > f_max - f_min`.

**Solution**:

Use a grid where adjacent bin pairs exist:

```python
HilbertParams(f_min=50, f_max=150, f_step=10)
```

### Issue: expected smoothing window is unavailable downstream

**Problem**: `condition_test` or `regression` requested `hilbert_smoothing_window_ms` that was not written by Hilbert.

**Solution**:

Make sure the Hilbert run included that window:

```python
HilbertParams(..., smoothing_windows_ms=[0, 250])
RegressionParams(..., hilbert_smoothing_window_ms=250)
```

### Issue: events are shifted or missing

**Problem**: events were read from the wrong source or need compatibility offset.

**Solution**:

- Use `events_source="events_tsv"` when relying on BIDS `_events.tsv`.
- Use `events_source="auto"` to prefer TSV and fall back to annotations.
- Use `event_sample_shift_samples` only when reproducing an external sample convention.

## Trial Statistics

### Issue: resolver labels do not match params

**Problem**: `TableTrialResolver.conditions[].label` differs from `condition_a` / `condition_b`.

**Solution**:

Keep labels identical:

```python
resolver = TableTrialResolver(conditions=[
    {"label": "condition_one", "when": {"column": "condition_code", "op": "==", "value": "1"}},
    {"label": "condition_two", "when": {"column": "condition_code", "op": "==", "value": "0"}},
])
params = ConditionTestParams(..., condition_a="condition_one", condition_b="condition_two")
```

### Issue: all statistics are NaN

**Common causes**:

- too few kept trials per condition;
- resolver conditions do not match behavior rows;
- epochs are dropped because they extend outside recording bounds;
- predictor values are missing or non-numeric in regression;
- epoch-cleaning settings are too strict.

**Solution**:

1. Inspect the companion `*_trials.tsv`.
2. Temporarily reduce filters and cleaning.
3. Confirm anchor event codes exist.
4. Confirm `min_trials_per_condition`.

### Issue: `window_ms` and `n_bins` error

**Problem**: both binning modes are enabled.

**Solution**:

Set only one:

```python
ConditionTestParams(..., window_ms=100, n_bins=0)
RegressionParams(..., window_ms=0, n_bins=24)
```

## Group Pipelines

### Issue: no compatible groups are created

**Problem**: subject-level files differ in condition labels, time axis, binning, or other signature fields.

**Solution**:

- Use the compatibility helper for the pipeline:

```python
build_condition_test_compatible_groups(files, primary_condition_metric="mean_difference")
build_regression_compatible_groups(files)
```

- Check that all subject-level files came from the same parameter set.
- Query a narrower set of files with `task`, `desc`, `run`, or `extension`.

### Issue: manual ROI channels are missing

**Problem**: channel names in `manual_region_channels` do not match subject-level output channels.

**Solution**:

- Check whether subject-level files used mono or bipolar channel names.
- Check case, separators, and contact naming.
- Start with one ROI and one subject, then scale up.

### Issue: atlas mode fails

**Problem**: subject-level provenance cannot find electrodes files or the atlas column is missing.

**Solution**:

- Include electrodes tables as secondary files in the subject-level run.
- Confirm the electrodes table has a channel column and the requested `atlas_name` column.
- Prefer channel-level subject outputs before group-level atlas ROI aggregation.

## Writers and Outputs

### Issue: output path description is unexpected

**Problem**: `output_description` becomes the BIDS `desc-...` entity.

**Solution**:

Set it explicitly:

```python
RegressionWriterParams(bids_root=ds.root, output_description="score")
```

### Issue: existing files are reprocessed

**Problem**: `skip_existing` is disabled or the predicted path differs from existing output.

**Solution**:

Use:

```python
processor.run(groups, writer, n_jobs=1, skip_existing=True)
```

Check that writer parameters, especially `output_description` and `output_format`, match previous runs.

## Before Reporting a Bug

Include:

- operating system and shell;
- `python -c "import sys; print(sys.executable)"`;
- `python -m pip show bidsforge`;
- exact script or Python snippet;
- full traceback;
- relevant BIDS tree excerpt;
- parameter objects and writer objects used for the run;
- the companion `*_trials.tsv` when statistics look wrong.

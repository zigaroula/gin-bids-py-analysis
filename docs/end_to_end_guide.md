# End-to-End Guide

Use this guide to get from a BIDS dataset to analysis derivatives with `bidsforge`. It is intentionally practical: use it first, then open the pipeline guide for the parameters you need to tune.

## 1. Install

Use a project-local virtual environment:

```bash
cd C:\GRE\dev\gin-bids-py-analysis
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Optional sanity check:

```bash
python -c "import bidsforge; print(bidsforge.__name__)"
```

## 2. Prepare Your BIDS Dataset

`bidsforge` expects BIDS-style files that can be queried by entities. A minimal input for Hilbert usually looks like:

```text
D:/data/bids/
|-- dataset_description.json
|-- sub-01/
|   `-- ieeg/
|       |-- sub-01_task-example_ieeg.vhdr
|       |-- sub-01_task-example_ieeg.vmrk
|       `-- sub-01_task-example_ieeg.eeg
`-- sub-02/
    `-- ...
```

For trial statistics, add the files used by the resolver and optional ROI mapping:

```text
D:/data/bids/
|-- sub-01/
|   |-- beh/
|   |   `-- sub-01_task-example_beh.tsv
|   `-- ieeg/
|       |-- sub-01_task-example_events.tsv
|       `-- sub-01_electrodes.tsv
`-- derivatives/
    `-- hilbert/
        `-- sub-01/
            `-- ieeg/
                `-- sub-01_task-example_desc-bandenv_ieeg.h5
```

## 3. Query Input Files

All pipelines start by creating a `BIDSDataset` and selecting files by BIDS entities:

```python
from bidsforge.bids import BIDSDataset

ds = BIDSDataset("D:/data/bids")
raw_ieeg = ds.get_files(scope="raw", suffix="ieeg", extension=".vhdr")
hilbert_outputs = ds.get_files(
    scope="hilbert",
    suffix="ieeg",
    extension=".h5",
    desc="bandenv",
)
```

For subject-level pipelines that need behavior/electrodes/events tables, use `build_subject_groups` so each subject is processed with its related files:

```python
from bidsforge.bids import build_subject_groups

groups = build_subject_groups(
    ds,
    ieeg_filters={"scope": "hilbert", "suffix": "ieeg", "extension": ".h5", "desc": "bandenv"},
    secondary_filters=[
        {"scope": "raw", "datatype": "beh", "suffix": "beh", "extension": ".tsv"},
        {"scope": "raw", "datatype": "ieeg", "suffix": "electrodes", "extension": ".tsv"},
    ],
)
```

## 4. Run Hilbert

Hilbert converts raw iEEG into band-envelope derivatives. This is the usual first analysis step:

```python
from bidsforge.processing.hilbert import (
    HilbertParams,
    HilbertProcessing,
    HilbertProcessingWriter,
    HilbertWriterParams,
)

params = HilbertParams(
    f_min=50,
    f_max=150,
    f_step=10,
    downsampled_frequency_hz=100,
    smoothing_windows_ms=[0, 250],
    normalization_mode="percent",
)
writer = HilbertProcessingWriter(
    HilbertWriterParams(
        bids_root=ds.root,
        output_format="hdf5",
        output_description="bandenv",
    )
)
paths = HilbertProcessing(params).run(raw_ieeg, writer, n_jobs=1, skip_existing=True)
```

Expected output:

```text
D:/data/bids/derivatives/hilbert/
|-- dataset_description.json
`-- sub-01/
    `-- ieeg/
        `-- sub-01_task-example_desc-bandenv_ieeg.h5
```

For full details, see [pipelines/hilbert.md](./pipelines/hilbert.md).

## 5. Run Subject-Level Statistics

Use `condition_test` when you compare two labels, such as condition one vs condition two trials:

```python
from bidsforge.processing.trial_stats import TableTrialResolver
from bidsforge.processing.trial_stats.condition_test import (
    ConditionTestParams,
    ConditionTestProcessing,
    ConditionTestProcessingWriter,
    ConditionTestWriterParams,
)

resolver = TableTrialResolver(
    conditions=[
        {"label": "condition_one", "when": {"column": "condition_code", "op": "==", "value": "1"}},
        {"label": "condition_two", "when": {"column": "condition_code", "op": "==", "value": "0"}},
    ],
)

params = ConditionTestParams(
    anchor_event_codes=["10"],
    tmin_s=-2.0,
    tmax_s=2.0,
    condition_a="condition_one",
    condition_b="condition_two",
    p_value_correction_method="fdr_bh",
)

writer = ConditionTestProcessingWriter(
    ConditionTestWriterParams(bids_root=ds.root, output_description="conditions")
)

ConditionTestProcessing(params, resolver=resolver).run(groups, writer, n_jobs=1)
```

Use `regression` when your question is about a continuous predictor, such as `score`:

```python
from bidsforge.processing.trial_stats.regression import (
    RegressionParams,
    RegressionProcessing,
    RegressionProcessingWriter,
    RegressionWriterParams,
)

resolver = TableTrialResolver(
    conditions=[
        {"label": "condition_one", "when": {"column": "condition_code", "op": "==", "value": "1"}},
        {"label": "condition_two", "when": {"column": "condition_code", "op": "==", "value": "2"}},
    ],
    extract_columns=["score", "response_time"],
)

params = RegressionParams(
    anchor_event_codes=["11", "12"],
    tmin_s=-1.0,
    tmax_s=6.0,
    condition_a="condition_one",
    condition_b="condition_two",
    predictor="score",
    activity_zscore="baseline",
)

writer = RegressionProcessingWriter(
    RegressionWriterParams(bids_root=ds.root, output_description="score")
)

RegressionProcessing(params, resolver=resolver).run(groups, writer, n_jobs=1)
```

Subject-level outputs include the main `.h5` or `.mat` statistics file and a companion `*_trials.tsv` audit table.

## 6. Run Group-Level ROI Statistics

Group pipelines consume subject-level outputs. First query the source files, then split them into compatible groups:

```python
from bidsforge.processing.trial_stats_group import (
    RegressionGroupParams,
    RegressionGroupProcessing,
    RegressionGroupProcessingWriter,
    RegressionGroupWriterParams,
    build_regression_compatible_groups,
)

files = ds.get_files(scope="regression", suffix="stats", extension=".h5", desc="score")
groups = build_regression_compatible_groups(files)

params = RegressionGroupParams(
    primary_regression_metric="slope",
    contrast_mode="paired",
    roi_mode="manual",
    manual_region_channels={
        "roi_a": {"01": ["A1", "A2"], "02": ["B1"]},
        "roi_b": {"01": ["C1"], "03": ["D1", "D2"]},
    },
    min_subjects_per_roi=2,
)

writer = RegressionGroupProcessingWriter(
    RegressionGroupWriterParams(bids_root=ds.root, output_description="scoregroup")
)

RegressionGroupProcessing(params).run(groups, writer, n_jobs=1)
```

For atlas-based ROI definitions, set `roi_mode="atlas"` and provide `atlas_name`, which must match a column in the electrodes tables referenced by the subject-level provenance.

## 7. Inspect or Visualize Outputs

Interactive visualization requires the `viz` extra:

```bash
python -m pip install -e ".[viz]"
```

Use `bidsforge.visualization.trial_stats` from your own project script or notebook to inspect saved results.

## 8. Recommended Workflow

1. Run one subject first.
2. Keep filters explicit: `scope`, `suffix`, `extension`, `desc`, and `task` when available.
3. Use `skip_existing=True` for resumable Hilbert or long subject-level runs.
4. Keep `output_description` meaningful because it becomes the BIDS `desc-...` entity.
5. Check the companion `*_trials.tsv` before interpreting statistics.
6. Run group statistics only after confirming that subject-level files are compatible.

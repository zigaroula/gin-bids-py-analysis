# gin-bids-py-analysis

BIDS-based iEEG analysis pipelines:
- `hilbert` (band envelope extraction)
- `delphos` (HFO/spike detection)
- `condition_test` (subject-level condition statistics)
- `regression` (subject-level condition-specific slope regression)
- `trial_stats_group` (group-level ROI statistics from `condition_test`)
- `trial_slope_stats_group` (group-level ROI statistics from `regression`)

## Installation (venv, by OS)

Requirements:
- Python `>=3.10`
- Git

### macOS / Linux (bash or zsh)

```bash
git clone <repo-url>
cd gin-bids-py-analysis

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

### Windows (PowerShell)

```powershell
git clone <repo-url>
cd gin-bids-py-analysis

py -3 -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -e .
```

If activation is blocked by execution policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### Windows (cmd.exe)

```bat
git clone <repo-url>
cd gin-bids-py-analysis

py -3 -m venv .venv
.venv\Scripts\activate.bat

python -m pip install --upgrade pip
python -m pip install -e .
```

Optional extras:

```bash
python -m pip install -e ".[viz]"   # visualization dependencies
python -m pip install -e ".[dev]"   # tests + viz tooling
```

Deactivate your environment with:

```bash
deactivate
```

## Pipeline System: General Process

All pipelines follow the same pattern:

1. Query BIDS files/groups (`BIDSDataset`, `get_files`, or `build_subject_groups`).
2. Configure algorithm parameters (`*Params` in each pipeline's `params.py`).
3. Configure output routing/format (`*WriterParams`).
4. Run processing (`*Processing.run(..., writer, n_jobs=...)`).
5. Read outputs in `derivatives/<pipeline_label>/...`.

The writer automatically creates a `dataset_description.json` in each derivatives pipeline folder.

Minimal pattern:

```python
from gin_bids_py_analysis.bids import BIDSDataset
from gin_bids_py_analysis.processing.hilbert import (
    HilbertParams, HilbertProcessing, HilbertWriterParams, HilbertProcessingWriter
)

ds = BIDSDataset("/path/to/bids")
files = ds.get_files(scope="raw", suffix="ieeg", extension=".vhdr")

processor = HilbertProcessing(HilbertParams(f_min=50, f_max=150, f_step=10))
writer = HilbertProcessingWriter(HilbertWriterParams(bids_root=ds.root))
out_paths = processor.run(files, writer, n_jobs=1)
```

## Typical Pipeline Order

1. `hilbert` or `delphos` from raw iEEG (`scope="raw"`).
2. `condition_test` from chosen iEEG derivatives (often Hilbert BrainVision outputs).
3. `regression` from chosen iEEG derivatives when your analysis is `gamma ~ continuous_value`.
4. `trial_stats_group` from `condition_test` outputs (`scope="condition_test"`).
5. `trial_slope_stats_group` from `regression` outputs (`scope="regression"`).

## Pipeline Details and Configuration

### 1) Hilbert (`scripts/run_hilbert.py`)

Run:

```bash
python scripts/run_hilbert.py
```

Main configuration is in `HilbertParams` (`src/gin_bids_py_analysis/processing/hilbert/params.py`):
- Frequency grid:
  - `f_min`, `f_max`, `f_step` (subbands are adjacent bin pairs).
  - Validation: `f_max > f_min`, and `f_step < (f_max - f_min)`.
- Downsampling:
  - `downsampled_frequency_hz` (`None` keeps original sampling rate).
- Montage:
  - `montage_mode` (`mono` or `bipolar`)
  - `bipolar_direction`, `bipolar_storage`
  - `channels_for_montage` and `channels_to_exclude_for_montage` (list or regex via `re.fullmatch`).
- Post-processing:
  - `smoothing_windows_ms` (`[0]` means no smoothing)
  - `normalization_mode` (`none`, `percent`, `percent_centered`, `db`)

Writer configuration (`HilbertWriterParams`):
- `output_format`: `"hdf5"` or `"brainvision"`
- `output_description`: used in `desc-...` entity

Important naming behavior:
- With BrainVision output, one file triplet is written per smoothing window.
- The writer appends `sm<window_ms>` to `desc`.
- Example: `output_description="gamma"` and window `0` -> `desc-gammasm0`.

This is why `condition_test` often filters with `desc: "gammasm0"`.

### 2) Delphos (`scripts/run_delphos.py`)

Run:

```bash
python scripts/run_delphos.py
```

Main configuration is in `DelphosParams` (`src/gin_bids_py_analysis/processing/delphos/params.py`):
- Detection controls:
  - `alpha`
  - `detection_type` (`["Osc"]`, `["Spk"]`, or both)
  - `freq_band` (list of `[low, high]` bands in Hz)
- Thresholding:
  - `thr_type` (`"auto"` or numeric)
  - `param_thr`
  - `quantile_method`
- Wavelet controls:
  - `nb_voices`
  - `vanishing_moment`
- Montage and channel selection:
  - same montage/channel selector pattern as Hilbert

Writer configuration (`DelphosWriterParams`):
- `output_format`: `"hdf5"` or `"tsv"`
- TSV mode writes:
  - `*_events.tsv` (event rows)
  - `*_rates.tsv` (per-channel event rates)

### 3) Condition Test (subject level) (`scripts/run_trial_stats.py`)

Run:

```bash
python scripts/run_trial_stats.py
```

This pipeline compares `condition_a` vs `condition_b` around anchor events, per subject.

Input selection and grouping:
- `IEEG_FILTERS`: selects iEEG files to analyze.
- `SECONDARY_FILTERS`: extra TSV/CSV files (behavior, electrodes, etc.).
- `build_subject_groups(...)`: builds one file group per subject.

Core configuration in `ConditionTestParams`
(`src/gin_bids_py_analysis/processing/trial_stats/condition_test/params.py`):
- Required:
  - `anchor_event_codes`
  - `tmin_s`, `tmax_s`
- Labels and minimum data:
  - `condition_a`, `condition_b`
  - `min_trials_per_condition`
  - `drop_partial_epochs`
- Statistical test:
  - `equal_var` (Welch if `False`)
  - `p_value_correction_method` (`none`, `fdr_bh`, `bonferroni`)
  - `significance_alpha`
- Feature space:
  - `atlas_name` and optional `atlas_regions` to run ROI-level stats instead of channel-level stats
- Temporal binning:
  - `window_ms` (fixed-width bins) or `n_bins` (count-based bins)
  - These are mutually exclusive (cannot both be active)

Resolver configuration (also critical):
- `TableTrialResolver` maps each anchor event to trial labels from tables.
- Most important settings:
  - `conditions`
  - `extract_columns`
  - optional onset/order/code/keep columns

Minimal examples:

```python
TableTrialResolver(
    conditions=[
        {"label": "accepted", "when": {"column": "choice", "op": "==", "value": "1"}},
        {"label": "rejected", "when": {"column": "choice", "op": "==", "value": "0"}},
    ]
)

TableTrialResolver(
    conditions=[
        {"label": "positive", "when": {"column": "rating", "op": ">", "value": 0}},
        {"label": "negative", "when": {"column": "rating", "op": "<", "value": 0}},
    ]
)

TableTrialResolver(
    conditions=[
        {
            "label": "accepted_high_conf",
            "when": {
                "all": [
                    {"column": "choice", "op": "==", "value": "1"},
                    {"column": "confidence", "op": ">=", "value": 4},
                ]
            },
        },
        {
            "label": "other",
            "when": {"any": [
                {"column": "choice", "op": "==", "value": "0"},
                {"column": "confidence", "op": "<", "value": 4},
            ]},
        },
    ]
)
```

Writer configuration (`ConditionTestWriterParams`):
- `output_format`: `"hdf5"` or `"matlab"`
- `include_epochs`: include per-trial epoch arrays (larger files)

Outputs:
- `*_stats.h5` or `*_stats.mat`
- plus a companion `*_trials.tsv` audit table
- default derivatives target: `derivatives/condition_test/...`
- default subject `desc`: `conditiontest`

Note for downstream `trial_stats_group`:
- `trial_stats_group` requires channel-level `condition_test` inputs.
- If you plan to run group stats later, keep `atlas_name=None` in `condition_test`.

### 4) Regression (subject level) (`scripts/run_trial_slope_stats.py`)

Run:

```bash
python scripts/run_trial_slope_stats.py
```

This pipeline computes, for each condition separately, a linear regression
between epoched gamma activity and a continuous predictor value coming from TSV/CSV.

Core configuration in `RegressionParams`
(`src/gin_bids_py_analysis/processing/trial_stats/regression/params.py`):
- Required:
  - `anchor_event_codes`
  - `tmin_s`, `tmax_s`
  - `predictor` (metadata key carrying the numeric value)
- Labels and minimum data:
  - `condition_a`, `condition_b`
  - `min_trials_per_condition` (default `3`)
  - `drop_partial_epochs`
- Statistical controls:
  - `p_value_correction_method` (`none`, `fdr_bh`, `bonferroni`)
  - `significance_alpha`
  - `predictor_zscore` (`none`, `condition`, `global`)
- Feature space and binning:
  - same `atlas_name` / `atlas_regions` behavior as `condition_test`
  - same `window_ms` / `n_bins` mutual exclusivity

Resolver configuration:
- Use `TableTrialResolver` as in `condition_test`.
- To inject the predictor from table columns into trial metadata, use:
  - `extract_columns=["<column_name>"]`.
- Trials with missing/non-numeric predictor values are excluded and tagged
  with `invalid_predictor_value`.

Writer configuration (`RegressionWriterParams`):
- `output_format`: `"hdf5"` or `"matlab"`
- `include_epochs`: optional per-trial epoch arrays

Outputs:
- `*_stats.h5` or `*_stats.mat`
- companion `*_trials.tsv` with predictor audit columns (`predictor_raw`, `predictor_value`)
- default derivatives target: `derivatives/regression/...`
- default subject `desc`: `regression`

### 5) Trial Stats Group (group level ROI) (`scripts/run_trial_stats_group.py`)

Run:

```bash
python scripts/run_trial_stats_group.py
```

This pipeline consumes many `condition_test` outputs and performs one-sample ROI statistics against 0 over time.

Input discovery:
- `TRIAL_STATS_FILTERS` should point to your `condition_test` derivatives (often `.h5`).
- `build_trial_stats_compatible_groups(...)` automatically splits files into compatible sets (same task, condition labels, time axis, and binning signature).

Core configuration in `TrialStatsGroupParams` (`src/gin_bids_py_analysis/processing/trial_stats_group/params.py`):
- Metric to aggregate from each subject file:
  - `source_metric`: `mean_difference`, `t_values`, `condition_a_mean`, `condition_b_mean`
- Statistical controls:
  - `p_value_correction_method`
  - `significance_alpha`
- ROI definition mode (exactly one mode per run):
  - `roi_mode="atlas"` with `atlas_name`
  - `roi_mode="manual"` with `manual_region_channels` shaped as:
    - `{roi_name: {subject_id: [channel_name, ...]}}`
- Inclusion thresholds:
  - `min_channels_per_roi`
  - `min_subjects_per_roi`

Writer configuration (`TrialStatsGroupWriterParams`):
- `output_format`: `"hdf5"` or `"matlab"`

## Visualization (brief)

There are two ways to visualize trial-stats results:

1. Interactive UI:
   - Script: `scripts/visualize_trial_stats.py`
   - Uses `ConditionTestParams` (ttest mode) or `RegressionParams` via `launch_slope(...)`.
   - Group tab supports both `trial_stats_group` and `trial_slope_stats_group`.
   - Requires viz dependencies (`pip install -e ".[viz]"`).

2. Static inspection:
   - Script: `scripts/inspect_trial_stats.py`
   - Reads subject-level `condition_test` / `regression` outputs or group-level outputs and produces a summary figure/console recap.

## Running Tests

```bash
pytest
pytest --cov=gin_bids_py_analysis --cov-report=term-missing
```

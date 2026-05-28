# Regression Pipeline

The `regression` pipeline performs subject-level trial-wise regression between iEEG activity and a continuous predictor.

## Objective

Use this pipeline when trial activity should be related to a numeric variable, such as a score, reaction time, confidence, or a model-derived value. Statistics are computed separately for `condition_a` and `condition_b`.

## Processing Summary

1. Load subject-level iEEG data from BrainVision or Hilbert `.h5/.mat` derivatives.
2. Resolve anchor events.
3. Resolve trial labels and predictor values from behavior tables.
4. Extract epochs and optionally apply trial annotators.
5. Optionally aggregate channels into atlas regions.
6. Clean epochs, z-score activity, transform/z-score predictors, and bin time.
7. Compute regression metrics such as slope and `r_value` per feature and time point.
8. Write statistics plus a companion trial audit table.

## Minimal Example

```python
from bidsforge.bids import BIDSDataset, build_subject_groups
from bidsforge.processing.trial_stats import TableTrialResolver
from bidsforge.processing.trial_stats.regression import (
    RegressionParams,
    RegressionProcessing,
    RegressionProcessingWriter,
    RegressionWriterParams,
)

ds = BIDSDataset("D:/data/bids")
groups = build_subject_groups(
    ds,
    {"scope": "hilbert", "suffix": "ieeg", "extension": ".h5", "desc": "bandenv"},
    [{"scope": "raw", "datatype": "beh", "suffix": "beh", "extension": ".tsv"}],
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
    min_trials_per_condition=3,
)

writer = RegressionProcessingWriter(
    RegressionWriterParams(bids_root=ds.root, output_description="score")
)

RegressionProcessing(params, resolver=resolver).run(groups, writer, n_jobs=1)
```

## Algorithm Parameters

`RegressionParams` includes the shared subject-level trial-statistics parameters plus regression-specific fields.

| Parameter | Default | Description |
| --- | --- | --- |
| `anchor_event_codes` | required | Event code or codes used as trial anchors. |
| `events_source` | `"annotations"` | `annotations`, `events_tsv`, or `auto`. |
| `event_sample_shift_samples` | `0` | Offset applied to anchor samples before epoch extraction. |
| `hilbert_smoothing_window_ms` | `0` | Smoothing window selected when reading Hilbert derivatives. |
| `tmin_s`, `tmax_s` | required | Epoch bounds in seconds relative to anchor. |
| `condition_a`, `condition_b` | `"condition_a"`, `"condition_b"` | Labels expected from the resolver. |
| `min_trials_per_condition` | `3` | Minimum kept trials in each condition. |
| `predictor` | `"predictor_value"` | Trial metadata key containing numeric predictor values. |
| `predictor_transform_by_condition` | `{}` | Optional affine transforms `{condition: {scale, offset}}`. |
| `predictor_zscore` | `"none"` | `none`, `condition`, or `global`. |
| `drop_partial_epochs` | `True` | Drop epochs extending outside recording bounds. |
| `p_value_correction_method` | `"fdr_bh"` | `none`, `fdr_bh`, or `bonferroni`. |
| `significance_alpha` | `0.05` | Threshold for corrected p-values. |
| `atlas_name` | `None` | Electrodes-table column used to group channels into ROIs. |
| `atlas_regions` | `[]` | Optional subset of atlas regions. Requires `atlas_name`. |
| `window_ms` | `0.0` | Non-overlapping temporal bin size in ms. |
| `n_bins` | `0` | Number of contiguous temporal bins. Mutually exclusive with `window_ms > 0`. |
| `activity_zscore` | `"none"` | `none`, `baseline`, or `across_trials`. |
| `activity_baseline_tmin_s`, `activity_baseline_tmax_s` | `-0.2`, `0.0` | Baseline window for `activity_zscore="baseline"`. |
| `activity_baseline_scope` | `"global"` | `trial`, `condition`, or `global`. |
| `experiment_start_event_code`, `experiment_end_event_code` | `None` | Optional bounds for anchor filtering. |
| `trial_activity_summary` | epoch mean | Per-trial activity summary saved in result files. |
| `epoch_cleaning` | disabled checks | Epoch/channel quality-control settings. |
| `n_permutations` | `0` | Number of label permutations to store for downstream cluster permutation. |
| `permutation_seed` | `None` | Random seed for subject-level permutations. |

## Predictor Handling

The predictor must be present in resolved trial metadata. With `TableTrialResolver`, add the predictor column to `extract_columns`:

```python
resolver = TableTrialResolver(
    conditions=[...],
    extract_columns=["score"],
)
params = RegressionParams(..., predictor="score")
```

Trials with missing or non-numeric predictor values are excluded from regression and annotated in the companion trial table.

Use `predictor_transform_by_condition` when signs or scales differ by condition:

```python
RegressionParams(
    ...,
    predictor="score",
    predictor_transform_by_condition={
        "condition_one": {"scale": 1.0, "offset": 0.0},
        "condition_two": {"scale": -1.0, "offset": 0.0},
    },
)
```

## Writer Parameters

`RegressionWriterParams`:

| Parameter | Default | Description |
| --- | --- | --- |
| `bids_root` | required | Root of the BIDS dataset. |
| `pipeline_label` | `"regression"` | Derivatives folder name. |
| `output_modality` | `"ieeg"` | BIDS datatype folder. |
| `output_suffix` | `"stats"` | BIDS suffix. |
| `output_description` | `"regression"` | Value used in `desc-<value>`. |
| `output_format` | `"hdf5"` | `hdf5` or `matlab`. |
| `include_epochs` | `False` | Store per-trial epoch arrays in the output file. |

## Outputs

```text
derivatives/regression/
|-- dataset_description.json
`-- sub-01/
    `-- ieeg/
        |-- sub-01_task-example_desc-score_stats.h5
        `-- sub-01_task-example_desc-score_trials.tsv
```

The trial table includes predictor audit columns such as raw and transformed predictor values.

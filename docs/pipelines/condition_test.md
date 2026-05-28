# Condition Test Pipeline

The `condition_test` pipeline performs subject-level statistics between two trial conditions.

## Objective

Use this pipeline when each trial belongs to one of two labels and you want to test activity differences over channels or atlas regions. A common example is condition one vs condition two trials around a decision event.

## Processing Summary

1. Load one or more iEEG files from a subject group.
2. Resolve anchor events from annotations or `_events.tsv`.
3. Use a `TrialResolver`, usually `TableTrialResolver`, to assign trial labels from behavior tables.
4. Extract epochs around anchor events.
5. Optionally aggregate channels into atlas regions.
6. Optionally z-score activity, clean epochs, and bin time.
7. Compute condition means, mean difference, t-values, p-values, and significance masks.
8. Write statistics plus a companion trial audit table.

## Minimal Example

```python
from bidsforge.bids import BIDSDataset, build_subject_groups
from bidsforge.processing.trial_stats import TableTrialResolver
from bidsforge.processing.trial_stats.condition_test import (
    ConditionTestParams,
    ConditionTestProcessing,
    ConditionTestProcessingWriter,
    ConditionTestWriterParams,
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
        {"label": "condition_two", "when": {"column": "condition_code", "op": "==", "value": "0"}},
    ],
)

params = ConditionTestParams(
    anchor_event_codes=["10"],
    tmin_s=-2.0,
    tmax_s=2.0,
    condition_a="condition_one",
    condition_b="condition_two",
)

writer = ConditionTestProcessingWriter(
    ConditionTestWriterParams(bids_root=ds.root, output_description="conditions")
)

ConditionTestProcessing(params, resolver=resolver).run(groups, writer, n_jobs=1)
```

## Algorithm Parameters

`ConditionTestParams` includes the shared subject-level trial-statistics parameters plus condition-test-specific fields.

| Parameter | Default | Description |
| --- | --- | --- |
| `anchor_event_codes` | required | Event code or codes used as trial anchors. |
| `events_source` | `"annotations"` | `annotations`, `events_tsv`, or `auto`. |
| `event_sample_shift_samples` | `0` | Offset applied to anchor samples before epoch extraction. |
| `hilbert_smoothing_window_ms` | `0` | Smoothing window selected when reading Hilbert `.h5` or `.mat` derivatives. |
| `tmin_s`, `tmax_s` | required | Epoch bounds in seconds relative to anchor. |
| `condition_a`, `condition_b` | `"condition_a"`, `"condition_b"` | Labels expected from the resolver. |
| `min_trials_per_condition` | `2` | Minimum kept trials in each condition. |
| `drop_partial_epochs` | `True` | Drop epochs extending outside recording bounds. |
| `equal_var` | `False` | Forwarded to `scipy.stats.ttest_ind`; `False` uses Welch's t-test. |
| `p_value_correction_method` | `"fdr_bh"` | `none`, `fdr_bh`, or `bonferroni`. |
| `significance_alpha` | `0.05` | Threshold for corrected p-values. |
| `atlas_name` | `None` | Electrodes-table column used to group channels into ROIs. |
| `atlas_regions` | `[]` | Optional subset of atlas regions. Requires `atlas_name`. |
| `window_ms` | `0.0` | Non-overlapping temporal bin size in ms. |
| `n_bins` | `0` | Number of contiguous temporal bins. Mutually exclusive with `window_ms > 0`. |
| `activity_zscore` | `"none"` | `none` or `baseline`. |
| `activity_baseline_tmin_s`, `activity_baseline_tmax_s` | `-0.2`, `0.0` | Baseline window for `activity_zscore="baseline"`. |
| `activity_baseline_scope` | `"global"` | `trial`, `condition`, or `global`. |
| `experiment_start_event_code`, `experiment_end_event_code` | `None` | Optional bounds for anchor filtering. |
| `trial_activity_summary` | epoch mean | Per-trial summary saved in result files. |
| `epoch_cleaning` | disabled checks | Optional epoch/channel quality-control settings. |
| `n_permutations` | `0` | Number of label permutations to store for downstream cluster permutation. |
| `permutation_seed` | `None` | Random seed for subject-level permutations. |
| `channel_significance_mode` | `"none"` | `none`, `single_bin`, or `duration`. |
| `channel_significance_duration_threshold_ms` | `100.0` | Duration threshold for `duration` channel significance. |

Validation notes:

- `anchor_event_codes` must not be empty.
- `tmax_s` must be greater than `tmin_s`.
- `condition_a` and `condition_b` must be different.
- `window_ms` and `n_bins` cannot both be active.
- Baseline windows must sit inside the epoch when `activity_zscore="baseline"`.

## Resolver Parameters

`TableTrialResolver` is the task-specific layer. It maps anchor events to rows in TSV/CSV files and assigns labels.

Common fields:

| Parameter | Description |
| --- | --- |
| `conditions` | Two condition definitions with `label` and `when`. Labels must match `condition_a` and `condition_b`. |
| `extract_columns` | Optional behavior columns copied into trial metadata. |
| `filter` | Optional BIDS entity filter for selecting source tables from a file group. |

Example with compound rules:

```python
TableTrialResolver(
    conditions=[
        {
            "label": "condition_one_high_conf",
            "when": {
                "all": [
                    {"column": "condition_code", "op": "==", "value": "1"},
                    {"column": "confidence", "op": ">=", "value": 4},
                ]
            },
        },
        {
            "label": "other",
            "when": {"column": "condition_code", "op": "==", "value": "0"},
        },
    ],
)
```

## Writer Parameters

`ConditionTestWriterParams`:

| Parameter | Default | Description |
| --- | --- | --- |
| `bids_root` | required | Root of the BIDS dataset. |
| `pipeline_label` | `"condition_test"` | Derivatives folder name. |
| `output_modality` | `"ieeg"` | BIDS datatype folder. |
| `output_suffix` | `"stats"` | BIDS suffix. |
| `output_description` | `"conditiontest"` | Value used in `desc-<value>`. |
| `output_format` | `"hdf5"` | `hdf5` or `matlab`. |
| `include_epochs` | `False` | Store per-trial epoch arrays in the output file. |

## Outputs

```text
derivatives/condition_test/
|-- dataset_description.json
`-- sub-01/
    `-- ieeg/
        |-- sub-01_task-example_desc-conditions_stats.h5
        `-- sub-01_task-example_desc-conditions_trials.tsv
```

For downstream `condition_test_group`, produce channel-level subject files by leaving `atlas_name=None`. Group-level ROI mapping is then applied consistently across subjects.

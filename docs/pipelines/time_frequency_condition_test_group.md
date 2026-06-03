# Time-Frequency Condition Test Group Pipeline

The `time_frequency_condition_test_group` pipeline aggregates channel-level
`time_frequency_condition_test` outputs into group-level ROI statistics on
time-frequency maps.

## Objective

Use this pipeline after subject-level `time_frequency_condition_test` when you
want ROI-level maps comparable to MATLAB `b3_TF_group_parcel_levels.m`.

The default metric is `t_values`, because the MATLAB b3 script loads each
subject/channel contrast t-map and displays the ROI mean over selected contacts.

## Processing Summary

1. Query subject-level `time_frequency_condition_test` stats files.
2. Split them into compatible groups with `build_time_frequency_condition_test_compatible_groups`.
3. Resolve ROI contributions from atlas metadata or manual channel maps.
4. Extract the requested channel-level TF metric from each subject file.
5. Apply inclusion thresholds by channel count and subject count.
6. Compute the ROI mean/SEM and a group one-sample t-test vs zero at each frequency/time point.
7. Optionally apply 2-D TF cluster correction.
8. Write group statistics under `derivatives/time_frequency_condition_test_group/`.

The output maps use:

```text
region x frequency x time
```

## Minimal Example

```python
from bidsforge.bids import BIDSDataset
from bidsforge.processing.time_frequency_stats_group.condition_test import (
    TimeFrequencyConditionTestGroupParams,
    TimeFrequencyConditionTestGroupProcessing,
    TimeFrequencyConditionTestGroupWriter,
    TimeFrequencyConditionTestGroupWriterParams,
    build_time_frequency_condition_test_compatible_groups,
)

ds = BIDSDataset("D:/data/bids")
files = ds.get_files(
    scope="time_frequency_condition_test",
    datatype="ieeg",
    suffix="stats",
    extension=".h5",
    desc="tfconditiontestonset",
)
groups = build_time_frequency_condition_test_compatible_groups(
    files,
    primary_condition_metric="t_values",
)

params = TimeFrequencyConditionTestGroupParams(
    primary_condition_metric="t_values",
    roi_mode="manual",
    manual_region_channels={
        "roi_a": {"01": ["A1", "A2"], "02": ["B1"]},
        "roi_b": {"01": ["C1"], "03": ["D1"]},
    },
    min_subjects_per_roi=2,
)

writer = TimeFrequencyConditionTestGroupWriter(
    TimeFrequencyConditionTestGroupWriterParams(
        bids_root=ds.root,
        output_description="tfconditiontestgroup",
    )
)

TimeFrequencyConditionTestGroupProcessing(params).run(groups, writer, n_jobs=1)
```

See `scripts/run_time_frequency_condition_test_group.py` for an editable runner.

## Algorithm Parameters

`TimeFrequencyConditionTestGroupParams`:

| Parameter | Default | Description |
| --- | --- | --- |
| `primary_condition_metric` | `"t_values"` | Metric read from each subject file: `t_values`, `mean_difference`, `condition_a_mean`, or `condition_b_mean`. |
| `p_value_correction_method` | `"none"` | `none`, `fdr_bh`, `bonferroni`, or `cluster_permutation`. |
| `significance_alpha` | `0.05` | Threshold used for significance masks. |
| `roi_mode` | required | `atlas` or `manual`. |
| `atlas_name` | `None` | Required when `roi_mode="atlas"`. Must match an electrodes-table column. |
| `manual_region_channels` | `{}` | Required when `roi_mode="manual"`. Shape: `{roi: {subject: [channels]}}`. |
| `min_channels_per_roi` | `1` | Minimum pooled channel count to keep a ROI. |
| `min_subjects_per_roi` | `1` | Minimum unique subject count to keep a ROI. |
| `cluster_permutation_method` | `"custom"` | `custom` uses stored subject permutations; `sign_flip` generates a group sign-flip null. |
| `cluster_threshold_alpha` | `0.05` | Cluster-forming threshold on the group p-map. |
| `cluster_percentile_alpha` | `0.005` | Signed percentile tail, matching MATLAB's 99.5/0.5 logic by default. |
| `n_group_permutations` | `10000` | Number of sign-flip permutations when `cluster_permutation_method="sign_flip"`. |
| `permutation_seed` | `None` | Random seed for sign-flip permutations. |

## ROI Modes

Atlas mode uses electrodes tables referenced by subject-level TF stats
provenance:

```python
TimeFrequencyConditionTestGroupParams(
    primary_condition_metric="t_values",
    roi_mode="atlas",
    atlas_name="MarsAtlas",
)
```

Manual mode is explicit and reproducible:

```python
TimeFrequencyConditionTestGroupParams(
    primary_condition_metric="t_values",
    roi_mode="manual",
    manual_region_channels={
        "roi_a": {"01": ["A1", "A2"], "02": ["B1"]},
    },
)
```

Subject identifiers are normalized in the same style as BIDS participant
labels, so both `"01"` and `"sub-01"` point to subject `01`.

## Cluster Correction

Cluster correction is disabled by default. When
`p_value_correction_method="cluster_permutation"`:

- clusters are formed on 2-D frequency/time maps with 8-connectivity;
- observed cluster statistics are sums of the ROI mean source map, not sums of
  the group t-map;
- `cluster_permutation_method="custom"` requires `permuted_t_values` in every
  subject-level condition-test file;
- `cluster_permutation_method="sign_flip"` is available for tests or exploratory
  use when subject-level permutations are absent.

The custom mode is the closest match to the older MATLAB b3 correction logic.

## Writer Parameters

`TimeFrequencyConditionTestGroupWriterParams`:

| Parameter | Default | Description |
| --- | --- | --- |
| `bids_root` | required | Root of the BIDS dataset. |
| `pipeline_label` | `"time_frequency_condition_test_group"` | Derivatives folder name. |
| `output_modality` | `"ieeg"` | BIDS datatype folder. |
| `output_suffix` | `"stats"` | BIDS suffix. |
| `output_description` | `"tfconditiontestgroup"` | Value used in `desc-<value>`. |
| `output_format` | `"hdf5"` | `hdf5` or `matlab`. |

## Outputs

```text
derivatives/time_frequency_condition_test_group/
|-- dataset_description.json
`-- sub-group/
    `-- ieeg/
        `-- sub-group_task-example_desc-tfconditiontestgroup_stats.h5
```

Main groups in the HDF5/MAT output:

| Group | Content |
| --- | --- |
| `axes` | `region`, `frequency_hz`, `time_s`. |
| `data/source_metric` | ROI mean and SEM of the selected source metric. |
| `data/signal_activity` | ROI condition mean and SEM maps. |
| `stats/source_metric` | Group t-values, p-values, corrected p-values, and significant masks. |
| `stats/source_metric_epoch` | One value per ROI after averaging each contribution over frequency/time. |
| `contributions` | ROI contribution labels and per-channel TF maps. |
| `provenance` | Source subject stats files and electrodes files. |

## Compatibility Requirements

Input files must agree on condition labels, frequency axis, time axis,
`power_mode`, `time_selection`, `baseline_grand_average`, task, source `desc`,
and selected metric. Use
`build_time_frequency_condition_test_compatible_groups` rather than manually
batching files; it separates incompatible inputs into independent group runs.

## Debug Comparison

Use `scripts/debug/compare_b3_TF_condition_test.py` to inspect Python group
outputs and optionally compare them to a numeric MATLAB export.

The stock MATLAB `b3_TF_group_parcel_levels.m` mostly saves PNG figures. For a
direct numerical comparison, export a `.mat` containing a ROI map such as
`mean_source_t_values` or `dots`, plus optional `region`, `frequency_hz`, and
`time_s` axes.

# Time-Frequency Regression Group Pipeline

The `time_frequency_regression_group` pipeline aggregates channel-level
`time_frequency_regression` outputs into group-level ROI statistics on
time-frequency maps.

## Objective

Use this pipeline after subject-level `time_frequency_regression` when you want
ROI-level regression maps comparable to MATLAB
`b3_TF_group_parcel_levels_CB.m`.

The default metric is `t_values`, because the MATLAB b3 CB script displays the
mean of subject/channel regression t-maps inside each ROI. The pipeline can
also aggregate `slope` and `r_value` maps.

## Processing Summary

1. Query subject-level `time_frequency_regression` stats files.
2. Split them into compatible groups with `build_time_frequency_regression_compatible_groups`.
3. Resolve ROI contributions from atlas metadata or manual channel maps.
4. Extract the requested regression metric from each subject file.
5. Compute condition-wise ROI mean/SEM maps.
6. Compute vs-zero statistics for each condition and a paired condition contrast.
7. Optionally apply 2-D TF cluster correction to the condition contrast.
8. Write group statistics under `derivatives/time_frequency_regression_group/`.

The output maps use:

```text
region x frequency x time
```

## Minimal Example

```python
from bidsforge.bids import BIDSDataset
from bidsforge.processing.time_frequency_stats_group.regression import (
    TimeFrequencyRegressionGroupParams,
    TimeFrequencyRegressionGroupProcessing,
    TimeFrequencyRegressionGroupWriter,
    TimeFrequencyRegressionGroupWriterParams,
    build_time_frequency_regression_compatible_groups,
)

ds = BIDSDataset("D:/data/bids")
files = ds.get_files(
    scope="time_frequency_regression",
    datatype="ieeg",
    suffix="stats",
    extension=".h5",
    desc="tfregressiononset",
)
groups = build_time_frequency_regression_compatible_groups(
    files,
    primary_regression_metric="t_values",
)

params = TimeFrequencyRegressionGroupParams(
    primary_regression_metric="t_values",
    contrast_mode="paired",
    roi_mode="manual",
    manual_region_channels={
        "roi_a": {"01": ["A1", "A2"], "02": ["B1"]},
        "roi_b": {"01": ["C1"], "03": ["D1"]},
    },
    min_subjects_per_roi=2,
)

writer = TimeFrequencyRegressionGroupWriter(
    TimeFrequencyRegressionGroupWriterParams(
        bids_root=ds.root,
        output_description="tfregressiongroup",
    )
)

TimeFrequencyRegressionGroupProcessing(params).run(groups, writer, n_jobs=1)
```

See `scripts/run_time_frequency_regression_group.py` for an editable runner.

## Algorithm Parameters

`TimeFrequencyRegressionGroupParams`:

| Parameter | Default | Description |
| --- | --- | --- |
| `primary_regression_metric` | `"t_values"` | Metric read from each subject file: `t_values`, `slope`, or `r_value`. |
| `contrast_mode` | `"paired"` | Paired condition contrast. |
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

Validation note: custom cluster correction for TF regression requires
`primary_regression_metric="slope"`, because subject-level files store
`permuted_slopes`. For `t_values` and `r_value`, use no correction, FDR,
Bonferroni, or exploratory `sign_flip`.

## ROI Modes

Atlas mode uses electrodes tables referenced by subject-level TF stats
provenance:

```python
TimeFrequencyRegressionGroupParams(
    primary_regression_metric="t_values",
    roi_mode="atlas",
    atlas_name="MarsAtlas",
)
```

Manual mode is explicit and reproducible:

```python
TimeFrequencyRegressionGroupParams(
    primary_regression_metric="t_values",
    roi_mode="manual",
    manual_region_channels={
        "roi_a": {"01": ["A1", "A2"], "02": ["B1"]},
    },
)
```

For larger projects, keep ROI definitions in a project CSV or script and pass
the resulting mapping as `manual_region_channels`.

## Cluster Correction

Cluster correction is disabled by default. When
`p_value_correction_method="cluster_permutation"`:

- clusters are formed on 2-D frequency/time maps with 8-connectivity;
- observed cluster statistics are sums of the ROI mean source contrast map;
- custom mode expects stored subject-level `permuted_slopes`;
- sign-flip mode is available for tests or exploratory use without stored
  subject permutations.

The custom mode is intended for Matlab-faithful slope maps. The default b3 CB
comparison path (`primary_regression_metric="t_values"`) prioritizes matching
the displayed Matlab maps rather than custom cluster correction.

## Writer Parameters

`TimeFrequencyRegressionGroupWriterParams`:

| Parameter | Default | Description |
| --- | --- | --- |
| `bids_root` | required | Root of the BIDS dataset. |
| `pipeline_label` | `"time_frequency_regression_group"` | Derivatives folder name. |
| `output_modality` | `"ieeg"` | BIDS datatype folder. |
| `output_suffix` | `"stats"` | BIDS suffix. |
| `output_description` | `"tfregressiongroup"` | Value used in `desc-<value>`. |
| `output_format` | `"hdf5"` | `hdf5` or `matlab`. |

## Outputs

```text
derivatives/time_frequency_regression_group/
|-- dataset_description.json
`-- sub-group/
    `-- ieeg/
        `-- sub-group_task-example_desc-tfregressiongroup_stats.h5
```

Main groups in the HDF5/MAT output:

| Group | Content |
| --- | --- |
| `axes` | `region`, `frequency_hz`, `time_s`. |
| `data/regression/source_metric` | ROI mean and SEM of the selected regression metric for each condition. |
| `data/regression/r_value` | ROI mean and SEM of r-values for each condition. |
| `data/signal_activity` | ROI condition mean and SEM activity maps. |
| `stats/regression/condition_contrast` | Paired A/B group t-values, p-values, corrected p-values, and masks. |
| `stats/regression/<condition>_vs_zero` | One-sample group tests vs zero per condition. |
| `stats/regression/epoch_summary` | One value per ROI after averaging contribution contrasts over frequency/time. |
| `contributions` | ROI contribution labels and per-channel TF maps. |
| `provenance` | Source subject stats files and electrodes files. |

## Compatibility Requirements

Input files must agree on condition labels, frequency axis, time axis,
`power_mode`, `time_selection`, `baseline_grand_average`, task, source `desc`,
selected metric, predictor, predictor z-score mode, and predictor transforms.
Use `build_time_frequency_regression_compatible_groups` rather than manually
batching files; it separates incompatible inputs into independent group runs.

## Debug Comparison

Use `scripts/debug/compare_b3_TF_regression.py` to inspect Python group outputs
and optionally compare them to a numeric MATLAB export.

The stock MATLAB `b3_TF_group_parcel_levels_CB.m` mostly saves PNG figures. For
a direct numerical comparison, export a `.mat` containing maps such as
`condition_a_mean_source_t_values` and `condition_b_mean_source_t_values`, or
fields named after MATLAB regressors such as `P_Rating` and `UP_Rating`.

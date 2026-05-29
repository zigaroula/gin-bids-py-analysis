# Regression Group Pipeline

The `regression_group` pipeline aggregates subject-level `regression` outputs into group-level ROI statistics.

## Objective

Use this pipeline after subject-level regression to test condition-specific regression metrics and condition contrasts across subjects or pooled ROI channel contributions.

## Processing Summary

1. Query subject-level `regression` stats files.
2. Split them into compatible groups with `build_regression_compatible_groups`.
3. Resolve ROI contributions from atlas metadata or manual channel maps.
4. Extract the requested regression metric from each subject file.
5. Compute condition-wise vs-zero statistics and condition contrasts.
6. Optionally apply multiple-comparison or cluster-permutation correction.
7. Write group statistics under `derivatives/regression_group/`.

## Minimal Example

```python
from bidsforge.bids import BIDSDataset
from bidsforge.processing.trial_stats_group import (
    RegressionGroupParams,
    RegressionGroupProcessing,
    RegressionGroupProcessingWriter,
    RegressionGroupWriterParams,
    build_regression_compatible_groups,
)

ds = BIDSDataset("D:/data/bids")
files = ds.get_files(scope="regression", suffix="stats", extension=".h5", desc="score")
groups = build_regression_compatible_groups(files)

params = RegressionGroupParams(
    primary_regression_metric="slope",
    contrast_mode="paired",
    roi_mode="manual",
    manual_region_channels={
        "roi_a": {"01": ["A1", "A2"], "02": ["B1"]},
        "roi_b": {"01": ["C1"], "03": ["D1"]},
    },
    min_subjects_per_roi=2,
)

writer = RegressionGroupProcessingWriter(
    RegressionGroupWriterParams(bids_root=ds.root, output_description="scoregroup")
)

RegressionGroupProcessing(params).run(groups, writer, n_jobs=1)
```

## Algorithm Parameters

`RegressionGroupParams`:

| Parameter | Default | Description |
| --- | --- | --- |
| `primary_regression_metric` | `"slope"` | Metric read from each subject file: `slope` or `r_value`. |
| `contrast_mode` | `"paired"` | `paired` or `unpaired` condition contrast. |
| `p_value_correction_method` | `"none"` | `none`, `fdr_bh`, `bonferroni`, or `cluster_permutation`. |
| `significance_alpha` | `0.05` | Threshold used for significance masks. |
| `roi_mode` | required | `atlas` or `manual`. |
| `atlas_name` | `None` | Required when `roi_mode="atlas"`. Must match an electrodes-table column. |
| `manual_region_channels` | `{}` | Required when `roi_mode="manual"`. Shape: `{roi: {subject: [channels]}}`. |
| `min_channels_per_roi` | `1` | Minimum pooled channel count to keep a ROI. |
| `min_subjects_per_roi` | `1` | Minimum unique subject count to keep a ROI. |
| `n_group_permutations` | `10000` | Null iterations for cluster permutation. |
| `cluster_threshold_alpha` | `0.05` | Within-iteration cluster-forming threshold. |
| `permutation_seed` | `None` | Random seed for group permutations. |
| `cluster_permutation_method` | `"custom"` | `custom` uses stored subject permutations; `sign_flip` uses sign-flip testing. |
| `n_clusters_to_keep` | `1` | Number of largest temporal clusters to test per ROI. |

Validation note: `cluster_permutation` is only supported with `contrast_mode="paired"`.

## ROI Modes

Atlas mode uses electrodes tables referenced by subject-level provenance:

```python
RegressionGroupParams(
    primary_regression_metric="slope",
    roi_mode="atlas",
    atlas_name="ExampleAtlas",
)
```

Manual mode is explicit and reproducible:

```python
RegressionGroupParams(
    primary_regression_metric="slope",
    roi_mode="manual",
    manual_region_channels={
        "roi_a": {"01": ["A1", "A2"], "02": ["B1"]},
    },
)
```

For larger projects, editable scripts under `scripts/` can contain helper functions for loading ROI definitions from CSV files and combining ROIs before group analysis.

## Writer Parameters

`RegressionGroupWriterParams`:

| Parameter | Default | Description |
| --- | --- | --- |
| `bids_root` | required | Root of the BIDS dataset. |
| `pipeline_label` | `"regression_group"` | Derivatives folder name. |
| `output_modality` | `"ieeg"` | BIDS datatype folder. |
| `output_suffix` | `"stats"` | BIDS suffix. |
| `output_description` | `"regressiongroup"` | Value used in `desc-<value>`. |
| `output_format` | `"hdf5"` | `hdf5` or `matlab`. |

## Outputs

```text
derivatives/regression_group/
|-- dataset_description.json
`-- ieeg/
    `-- task-example_desc-scoregroup_stats.h5
```

The exact path depends on the entities shared by the compatible input group.

## Compatibility Requirements

Input files must agree on condition labels, predictor metadata, time axis, binning, and analysis structure. Use `build_regression_compatible_groups(files)` so incompatible files are separated automatically.

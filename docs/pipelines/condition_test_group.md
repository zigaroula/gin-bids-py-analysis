# Condition Test Group Pipeline

The `condition_test_group` pipeline aggregates channel-level `condition_test` outputs into group-level ROI statistics.

## Objective

Use this pipeline after subject-level `condition_test` when you want to test whether a selected condition metric differs from zero across pooled ROI channel contributions.

## Processing Summary

1. Query subject-level `condition_test` stats files.
2. Split them into compatible groups with `build_condition_test_compatible_groups`.
3. Resolve ROI contributions from atlas metadata or manual channel maps.
4. Extract the requested condition metric from each subject file.
5. Apply inclusion thresholds by channel count and subject count.
6. Compute group-level one-sample statistics over time.
7. Write group statistics under `derivatives/condition_test_group/`.

## Minimal Example

```python
from bidsforge.bids import BIDSDataset
from bidsforge.processing.trial_stats_group import (
    ConditionTestGroupParams,
    ConditionTestGroupProcessing,
    ConditionTestGroupProcessingWriter,
    ConditionTestGroupWriterParams,
    build_condition_test_compatible_groups,
)

ds = BIDSDataset("D:/data/bids")
files = ds.get_files(scope="condition_test", suffix="stats", extension=".h5", desc="conditions")
groups = build_condition_test_compatible_groups(
    files,
    primary_condition_metric="mean_difference",
)

params = ConditionTestGroupParams(
    primary_condition_metric="mean_difference",
    roi_mode="manual",
    manual_region_channels={
        "roi_a": {"01": ["A1", "A2"], "02": ["B1"]},
        "roi_b": {"01": ["C1"], "03": ["D1"]},
    },
    min_subjects_per_roi=2,
)

writer = ConditionTestGroupProcessingWriter(
    ConditionTestGroupWriterParams(bids_root=ds.root, output_description="conditionsgroup")
)

ConditionTestGroupProcessing(params).run(groups, writer, n_jobs=1)
```

## Algorithm Parameters

`ConditionTestGroupParams`:

| Parameter | Default | Description |
| --- | --- | --- |
| `primary_condition_metric` | `"mean_difference"` | Metric read from each subject file: `mean_difference`, `t_values`, `condition_a_mean`, or `condition_b_mean`. |
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

## ROI Modes

Atlas mode:

```python
ConditionTestGroupParams(
    primary_condition_metric="mean_difference",
    roi_mode="atlas",
    atlas_name="ExampleAtlas",
)
```

Manual mode:

```python
ConditionTestGroupParams(
    primary_condition_metric="mean_difference",
    roi_mode="manual",
    manual_region_channels={
        "roi_a": {"01": ["A1", "A2"], "02": ["B1"]},
    },
)
```

Subject identifiers are normalized in the same style as BIDS participant labels, so both `"01"` and `"sub-01"` point to subject `01`.

## Writer Parameters

`ConditionTestGroupWriterParams`:

| Parameter | Default | Description |
| --- | --- | --- |
| `bids_root` | required | Root of the BIDS dataset. |
| `pipeline_label` | `"condition_test_group"` | Derivatives folder name. |
| `output_modality` | `"ieeg"` | BIDS datatype folder. |
| `output_suffix` | `"stats"` | BIDS suffix. |
| `output_description` | `"conditiontestgroup"` | Value used in `desc-<value>`. |
| `output_format` | `"hdf5"` | `hdf5` or `matlab`. |

## Outputs

```text
derivatives/condition_test_group/
|-- dataset_description.json
`-- ieeg/
    `-- task-example_desc-conditionsgroup_stats.h5
```

The exact path depends on the BIDS entities shared by the compatible input group.

## Compatibility Requirements

Input files must be compatible in task/run-level metadata, condition labels, analysis level, time axis, and binning signature. Use `build_condition_test_compatible_groups` rather than manually batching files; it separates incompatible inputs into independent group runs.

"""Run group-level ROI statistics on time-frequency condition-test outputs."""

from __future__ import annotations

from pathlib import Path

from bidsforge.bids import BIDSDataset, BIDSFile, BIDSFileGroup
from bidsforge.processing.time_frequency_stats_group import (
    TimeFrequencyConditionTestGroupParams,
    TimeFrequencyConditionTestGroupProcessing,
    TimeFrequencyConditionTestGroupWriter,
    TimeFrequencyConditionTestGroupWriterParams,
    build_time_frequency_condition_test_compatible_groups,
)


BIDS_ROOT = Path("path/to/bids_dataset")

STATS_FILTERS = {
    "scope": "time_frequency_condition_test",
    "datatype": "ieeg",
    "suffix": "stats",
    "extension": ".h5",
    "desc": "tfconditiontest",
}

N_JOBS = 1
SKIP_EXISTING = True

MANUAL_REGION_CHANNELS = {
    "example_roi": {
        "01": ["A1", "A2"],
        "02": ["A1", "A2"],
    },
}

PARAMS = TimeFrequencyConditionTestGroupParams(
    primary_condition_metric="t_values",
    p_value_correction_method="none",
    cluster_permutation_method="custom",
    significance_alpha=0.05,
    roi_mode="manual",
    manual_region_channels=MANUAL_REGION_CHANNELS,
    min_channels_per_roi=1,
    min_subjects_per_roi=1,
    n_group_permutations=1000,
    permutation_seed=1,
)

WRITER_PARAMS = TimeFrequencyConditionTestGroupWriterParams(
    bids_root=BIDS_ROOT,
    output_format="hdf5",
    output_description="tfconditiontestgroup",
)


def load_stats_files(dataset: BIDSDataset) -> list[BIDSFile]:
    return sorted(dataset.get_files(**STATS_FILTERS), key=lambda file: str(file.path))


def build_groups(files: list[BIDSFile]) -> list[BIDSFileGroup]:
    return build_time_frequency_condition_test_compatible_groups(
        files,
        primary_condition_metric=PARAMS.primary_condition_metric,
    )


def main() -> list[Path]:
    dataset = BIDSDataset(BIDS_ROOT)
    files = load_stats_files(dataset)
    groups = build_groups(files)
    print(f"Found {len(files)} TF condition-test file(s) in {len(groups)} group(s).")
    processor = TimeFrequencyConditionTestGroupProcessing(PARAMS)
    writer = TimeFrequencyConditionTestGroupWriter(WRITER_PARAMS)
    out_paths = processor.run(groups, writer, n_jobs=N_JOBS, skip_existing=SKIP_EXISTING)
    for path in out_paths:
        print(f"Wrote {path}")
    return out_paths


if __name__ == "__main__":
    main()

"""Run group-level ROI statistics on regression outputs.

Edit the configuration block below, then run from the repository root:

    python scripts/run_regression_group.py
"""

from __future__ import annotations

from pathlib import Path

from bidsforge.bids import BIDSDataset, BIDSFile, BIDSFileGroup
from bidsforge.processing.trial_stats_group import (
    RegressionGroupParams,
    RegressionGroupProcessing,
    RegressionGroupProcessingWriter,
    RegressionGroupWriterParams,
    build_regression_compatible_groups,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BIDS_ROOT = Path("path/to/bids_dataset")

# These filters should match subject-level files written by run_regression.py.
STATS_FILTERS = {
    "scope": "regression",
    "datatype": "ieeg",
    "suffix": "stats",
    "extension": ".h5",
    "desc": "regression",
}

N_JOBS = 1
SKIP_EXISTING = True

# Generic manual ROI example. Replace subject IDs and channel names with values
# from your dataset, or switch to roi_mode="atlas" and set atlas_name.
MANUAL_REGION_CHANNELS = {
    "example_roi": {
        "01": ["A1", "A2"],
        "02": ["A1", "A2"],
    },
}

PARAMS = RegressionGroupParams(
    # "slope" is the usual regression effect map; "r_value" is also available.
    primary_regression_metric="slope",
    contrast_mode="paired",
    p_value_correction_method="none",
    cluster_permutation_method="sign_flip",
    significance_alpha=0.05,
    roi_mode="manual",
    manual_region_channels=MANUAL_REGION_CHANNELS,
    # roi_mode="atlas",
    # atlas_name="your_atlas_column",
    min_channels_per_roi=1,
    min_subjects_per_roi=1,
    n_group_permutations=1000,
    permutation_seed=1,
)

# Writer parameters control where the group derivative is written under
# derivatives/regression_group.
WRITER_PARAMS = RegressionGroupWriterParams(
    bids_root=BIDS_ROOT,
    output_format="hdf5",
    output_description="regressiongroup",
)


def load_stats_files(dataset: BIDSDataset) -> list[BIDSFile]:
    return sorted(dataset.get_files(**STATS_FILTERS), key=lambda file: str(file.path))


def build_groups(files: list[BIDSFile]) -> list[BIDSFileGroup]:
    # Compatibility grouping avoids mixing files with different time axes,
    # conditions, predictor settings, or other metadata.
    return build_regression_compatible_groups(files)


def main() -> list[Path]:
    dataset = BIDSDataset(BIDS_ROOT)
    files = load_stats_files(dataset)
    groups = build_groups(files)
    print(
        f"Found {len(files)} regression file(s) "
        f"in {len(groups)} compatible group(s). Running with n_jobs={N_JOBS}."
    )

    processor = RegressionGroupProcessing(PARAMS)
    writer = RegressionGroupProcessingWriter(WRITER_PARAMS)
    out_paths = processor.run(groups, writer, n_jobs=N_JOBS, skip_existing=SKIP_EXISTING)

    for path in out_paths:
        print(f"Wrote {path}")
    return out_paths


if __name__ == "__main__":
    main()

"""
Group-level ROI statistics on regression outputs.

Edit the shared recipe in scripts/debug/trial_slope_shared.py, then run:

    python scripts/debug/run_trial_slope_stats_group.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from bidsforge.bids import BIDSDataset, BIDSFile, BIDSFileGroup
from bidsforge.processing.trial_stats_group import (
    RegressionGroupProcessing,
    RegressionGroupProcessingWriter,
    build_regression_compatible_groups,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from trial_slope_shared import (  # noqa: E402
    BIDS_ROOT,
    RECIPE,
    TRIAL_SLOPE_GROUP_N_JOBS,
    build_group_params,
    build_regression_group_writer_params,
    print_recipe_summary,
    print_roi_summary,
)


def _load_trial_slope_stats_files(dataset: BIDSDataset) -> list[BIDSFile]:
    files = dataset.get_files(**RECIPE.trial_slope_stats_filters())
    return sorted(files, key=lambda file: str(file.path))


def _build_groups(files: list[BIDSFile]) -> list[BIDSFileGroup]:
    return build_regression_compatible_groups(files)


def main() -> list[Path]:
    print_recipe_summary()
    params = build_group_params()
    print_roi_summary(params.manual_region_channels)

    ds = BIDSDataset(BIDS_ROOT)
    files = _load_trial_slope_stats_files(ds)
    groups = _build_groups(files)
    print(
        f"Found {len(files)} regression file(s) grouped into {len(groups)} "
        f"compatible run(s). Running with n_jobs={TRIAL_SLOPE_GROUP_N_JOBS}."
    )

    if not groups:
        return []

    processor = RegressionGroupProcessing(params)
    writer = RegressionGroupProcessingWriter(build_regression_group_writer_params())
    out_paths = processor.run(groups, writer, n_jobs=TRIAL_SLOPE_GROUP_N_JOBS)
    for path in out_paths:
        print(f"Wrote {path}")
    return out_paths


if __name__ == "__main__":
    main()


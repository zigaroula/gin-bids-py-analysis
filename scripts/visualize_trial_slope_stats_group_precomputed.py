"""
Launch the interactive group visualization for a pre-computed trial-slope result.

Edit the shared recipe in scripts/trial_slope_shared.py, then run:

    python scripts/visualize_trial_slope_stats_group_precomputed.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from gin_bids_py_analysis.bids import BIDSDataset
from gin_bids_py_analysis.visualization.trial_stats import launch_group_precomputed

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from trial_slope_shared import (  # noqa: E402
    BIDS_ROOT,
    GROUP_STATS_FILTERS,
    build_group_params,
    print_recipe_summary,
    print_roi_summary,
)


if __name__ == "__main__":
    print_recipe_summary()
    group_params = build_group_params()
    print_roi_summary(group_params.manual_region_channels)

    ds = BIDSDataset(BIDS_ROOT)
    group_stats_files = ds.get_files(**GROUP_STATS_FILTERS)
    if not group_stats_files:
        raise FileNotFoundError(
            f"No group stats file found in {BIDS_ROOT} matching {GROUP_STATS_FILTERS}"
        )

    launch_group_precomputed(
        group_stats_files[0].path,
        group_params=group_params,
    )


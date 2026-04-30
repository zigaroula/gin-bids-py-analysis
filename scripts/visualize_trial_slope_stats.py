"""
Trial slope statistics visualization script.
Edit parameters below, then run:

    python scripts/visualize_trial_slope_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from gin_bids_py_analysis.bids import BIDSDataset
from gin_bids_py_analysis.visualization.trial_stats import launch_slope

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from trial_slope_shared import (  # noqa: E402
    BIDS_ROOT,
    PARAMS,
    RESOLVER,
    VM_PFC_SPIKE_EXCLUSION_REASON,
    build_group_params,
    build_trial_annotators,
    build_trial_slope_subject_groups,
    count_excluded_trials_by_reason as _count_excluded_trials_by_reason,
    format_subject_trial_exclusion_summary as _format_subject_trial_exclusion_summary,
    load_roi_channels_from_csv,
    print_recipe_summary,
    print_roi_summary,
    print_subject_trial_exclusion_summary,
)


def main() -> None:
    print_recipe_summary()
    manual_region_channels = load_roi_channels_from_csv()
    group_params = build_group_params(manual_region_channels)
    print_roi_summary(group_params.manual_region_channels)
    annotators = build_trial_annotators(manual_region_channels)

    ds = BIDSDataset(BIDS_ROOT)
    subject_groups = build_trial_slope_subject_groups(ds)
    print(f"Found {len(subject_groups)} subject(s).")

    launch_slope(
        subject_groups,
        PARAMS,
        RESOLVER,
        group_params=group_params,
        bids_root=BIDS_ROOT,
        annotators=annotators,
        subject_result_callback=print_subject_trial_exclusion_summary,
    )


if __name__ == "__main__":
    main()

"""
Example script to launch the interactive visualization for pre-computed trial statistics results.
Edit BIDS_ROOT, IEEG_FILTERS, SECONDARY_FILTERS, PARAMS, and RESOLVER below,
then run:

    python scripts/visualize_trial_stats_precomputed.py
"""

from __future__ import annotations

from pathlib import Path

from bidsforge.bids import BIDSDataset
from bidsforge.processing.trial_stats_group import ConditionTestGroupParams
from bidsforge.visualization.trial_stats import launch_precomputed

# ---------------------------------------------------------------------------
# Parameters  (edit these)
# ---------------------------------------------------------------------------

BIDS_ROOT = Path(r"E:\CBT\bids")

# Optional: configure group-level ROI statistics.
# When set, a "Group" tab appears after all subjects have been computed.
# Set to None to disable the Group tab.
GROUP_PARAMS = ConditionTestGroupParams(
    primary_condition_metric="t_values",
    p_value_correction_method="cluster_permutation",
    cluster_permutation_method="mne",
    significance_alpha=0.05,
    # roi_mode="atlas",
    # atlas_name="MarsAtlas",
    roi_mode="manual",
    manual_region_channels={
        "daINS": {
            "epi01": ["Y02", "Y06"],
            "epi03": ["IAD2"],
            "epi05": ["X04", "X07", "X03", "X06"],
            "epi07": ["X08", "T03"],
            "epi11": ["X05", "X07", "X06"],
            "epi12": ["Ap02"],
            "epi14": ["Xp04"],
            "epi17": ["II8"],
            "epi18": ["X06"],
            "epi19": ["IMD2"],
            "epi21": ["XS7", "XD7", "XD4", "XD8"],
            "epi22": ["XS8"],
            "epi23": ["EL2"]
        },
        "vaINS": {
            "epi04": ["Bp02"],
            "epi08": ["IA2", "IA4"],
            "epi11": ["X04"],
            "epi14": ["Xp02"],
            "epi17": ["II2"],
            "epi18": ["Y02", "X02"],
            "epi21": ["XS4"],
            "epi22": ["XS2", "YS2"]
        },
    }
)

# Discover subject-level stats files written by ConditionTestProcessingWriter.
STATS_FILTERS = {
    "suffix": "stats",
    "extension": ".h5",
    "desc": "conditiontest",
    "scope": "condition_test"
}

GROUP_STATS_FILTERS = {
    "suffix": "stats",
    "extension": ".h5",
    "desc": "conditiontestgroup",
    "scope": "condition_test_group",
}

if __name__ == "__main__":
    ds = BIDSDataset(BIDS_ROOT)

    subject_stats_files = {
        f.get("subject"): f.path
        for f in ds.get_files(**STATS_FILTERS)
    }

    group_stats_file = None
    group_stats_files = ds.get_files(**GROUP_STATS_FILTERS)
    if group_stats_files:
        group_stats_file = group_stats_files[0].path

    launch_precomputed(
        subject_stats_files,
        group_stats_file=group_stats_file,
        group_params=GROUP_PARAMS,
    )



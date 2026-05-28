"""
Example script to launch the interactive group visualization for a pre-computed
trial statistics group result.

Edit BIDS_ROOT, GROUP_STATS_FILTERS, and GROUP_PARAMS below, then run:

    python scripts/visualize_trial_stats_group_precomputed.py
"""

from __future__ import annotations

from pathlib import Path

from bidsforge.bids import BIDSDataset
from bidsforge.processing.trial_stats_group import ConditionTestGroupParams
from bidsforge.visualization.trial_stats import launch_group_precomputed

# ---------------------------------------------------------------------------
# Parameters  (edit these)
# ---------------------------------------------------------------------------

BIDS_ROOT = Path(r"E:\CBT\bids")

# Optional: pre-populate the GroupParamsPanel with the parameters that were
# used to produce the group file.  Set to None to leave the panel at defaults.
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
            "epi23": ["EL2"],
        },
        "vaINS": {
            "epi04": ["Bp02"],
            "epi08": ["IA2", "IA4"],
            "epi11": ["X04"],
            "epi14": ["Xp02"],
            "epi17": ["II2"],
            "epi18": ["Y02", "X02"],
            "epi21": ["XS4"],
            "epi22": ["XS2", "YS2"],
        },
    },
)

# Filters to discover the group stats file written by
# ConditionTestGroupProcessingWriter.
GROUP_STATS_FILTERS = {
    "suffix": "stats",
    "extension": ".h5",
    "desc": "conditiontestgroup",
    "scope": "condition_test_group",
}

if __name__ == "__main__":
    ds = BIDSDataset(BIDS_ROOT)

    group_stats_files = ds.get_files(**GROUP_STATS_FILTERS)
    if not group_stats_files:
        raise FileNotFoundError(
            f"No group stats file found in {BIDS_ROOT} matching {GROUP_STATS_FILTERS}"
        )

    group_stats_file = group_stats_files[0].path

    launch_group_precomputed(
        group_stats_file,
        group_params=GROUP_PARAMS,
    )


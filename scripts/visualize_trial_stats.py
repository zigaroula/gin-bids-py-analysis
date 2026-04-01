"""
Trial statistics visualization script.
Edit BIDS_ROOT, IEEG_FILTERS, SECONDARY_FILTERS, PARAMS, and RESOLVER below,
then run:

    python scripts/visualize_trial_stats.py
"""

from __future__ import annotations

from pathlib import Path

from gin_bids_py_analysis.bids import BIDSDataset, BIDSFileGroup, build_subject_groups
from gin_bids_py_analysis.processing.trial_stats import (
    TableTrialLabelResolver,
    TrialStatsParams,
)
from gin_bids_py_analysis.processing.trial_stats_group import TrialStatsGroupParams
from gin_bids_py_analysis.visualization.trial_stats import launch

# ---------------------------------------------------------------------------
# Parameters  (edit these)
# ---------------------------------------------------------------------------

BIDS_ROOT = Path(r"E:\CBT\bids")

# iEEG files to visualize. These are grouped per subject.
IEEG_FILTERS = {
    "suffix": "ieeg",
    "extension": ".vhdr",
    "desc": "gammasm250",
}

# Optional secondary tables used by the task-specific resolver.
# Adjust these filters to match where your events / behaviour tables live.
SECONDARY_FILTERS = [
    {"scope": "raw", "datatype": "beh", "suffix": "beh", "extension": ".tsv"},
    {"scope": "raw", "datatype": "ieeg", "suffix": "electrodes", "extension": ".tsv"},
]

PARAMS = TrialStatsParams(
    anchor_event_codes=["10"],
    tmin_s=-2.0,
    tmax_s=2.0,
    condition_a="accepted",
    condition_b="rejected",
    # atlas_name="MarsAtlas",
    # n_bins=24,
    p_value_correction_method="fdr_bh",
    significance_alpha=0.05,
)

# Update the column names and label map to match your dataset.
RESOLVER = TableTrialLabelResolver(
    label_column="choice",
    label_map={
        "0": "rejected",
        "1": "accepted",
    },
)

# Optional: configure group-level ROI statistics.
# When set, a "Group" tab appears after all subjects have been computed.
# Set to None to disable the Group tab.
GROUP_PARAMS = TrialStatsGroupParams(
    source_metric="t_values",
    p_value_correction_method="none",
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
            "epi17": ["II08"],
            "epi18": ["X06"],
            "epi19": ["IMD2"],
            "epi21": ["XS07", "XD07", "XD04", "XD08"],
            "epi22": ["XS08"],
            "epi23": ["EL02"]
        },
        "vaINS": {
            "epi04": ["Bp2"],
            "epi08": ["IA2", "IA4"],
            "epi11": ["X04"],
            "epi14": ["Xp02"],
            "epi17": ["II02"],
            "epi18": ["Y02", "X02"],
            "epi21": ["XS04"],
            "epi22": ["XS02", "YS02"]
        },
    }
)


def _build_subject_groups(dataset: BIDSDataset) -> dict[str, BIDSFileGroup]:
    groups = build_subject_groups(dataset, IEEG_FILTERS, SECONDARY_FILTERS)
    return {group.primary.get("subject"): group for group in groups}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = BIDSDataset(BIDS_ROOT)
    subject_groups = _build_subject_groups(ds)
    print(f"Found {len(subject_groups)} subject(s).")
    launch(subject_groups, PARAMS, RESOLVER, group_params=GROUP_PARAMS, bids_root=BIDS_ROOT)

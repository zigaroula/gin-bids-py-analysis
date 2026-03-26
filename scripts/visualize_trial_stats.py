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
from gin_bids_py_analysis.visualization.trial_stats import launch

# ---------------------------------------------------------------------------
# Parameters  (edit these)
# ---------------------------------------------------------------------------

BIDS_ROOT = Path(r"D:\CBT\bids")

# iEEG files to visualize. These are grouped per subject.
IEEG_FILTERS = {
    "suffix": "ieeg",
    "extension": ".vhdr",
    "desc": "gammasm0",
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
    tmax_s=10.0,
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
    launch(subject_groups, PARAMS, RESOLVER)

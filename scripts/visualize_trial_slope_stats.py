"""
Trial slope statistics visualization script.
Edit BIDS_ROOT, IEEG_FILTERS, SECONDARY_FILTERS, PARAMS, and RESOLVER below,
then run:

    python scripts/visualize_trial_slope_stats.py
"""

from __future__ import annotations

from pathlib import Path

from gin_bids_py_analysis.bids import BIDSDataset, BIDSFileGroup, build_subject_groups
from gin_bids_py_analysis.processing.trial_slope_stats import TrialSlopeStatsParams
from gin_bids_py_analysis.processing.trial_stats import TableTrialLabelResolver
from gin_bids_py_analysis.visualization.trial_stats import launch_slope

BIDS_ROOT = Path(r"D:\data_clarissa\valuation\bids")

IEEG_FILTERS = {
    "suffix": "ieeg",
    "extension": ".vhdr",
    "desc": "gammasm250",
}

SECONDARY_FILTERS = [
    {"scope": "raw", "datatype": "beh", "suffix": "beh", "extension": ".tsv"},
    {"scope": "raw", "datatype": "ieeg", "suffix": "electrodes", "extension": ".tsv"},
]

PARAMS = TrialSlopeStatsParams(
    anchor_event_codes=["11", "12"],
    tmin_s=-1.0,
    tmax_s=6.0,
    condition_a="pleasant",
    condition_b="unpleasant",
    predictor_metadata_key="rating",
    p_value_correction_method="none",
    significance_alpha=0.05,
)

RESOLVER = TableTrialLabelResolver(
    label_column="pleasant",
    label_map={
        "1.0": "unpleasant",
        "2.0": "pleasant",
    },
    extra_metadata_columns={
        "rating": "rating",
    },
)


def _build_subject_groups(dataset: BIDSDataset) -> dict[str, BIDSFileGroup]:
    groups = build_subject_groups(dataset, IEEG_FILTERS, SECONDARY_FILTERS)
    return {group.primary.get("subject"): group for group in groups}


if __name__ == "__main__":
    ds = BIDSDataset(BIDS_ROOT)
    subject_groups = _build_subject_groups(ds)
    print(f"Found {len(subject_groups)} subject(s).")
    launch_slope(subject_groups, PARAMS, RESOLVER, bids_root=BIDS_ROOT)

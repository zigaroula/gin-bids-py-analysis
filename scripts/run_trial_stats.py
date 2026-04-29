"""
Trial statistics on iEEG recordings - run script.
Edit the parameters below and run: python scripts/run_trial_stats.py
"""

from __future__ import annotations

from pathlib import Path

from gin_bids_py_analysis.bids import BIDSDataset, BIDSFileGroup, build_subject_groups
from gin_bids_py_analysis.processing.trial_stats import TableTrialResolver
from gin_bids_py_analysis.processing.trial_stats.condition_test import (
    ConditionTestParams,
    ConditionTestProcessing,
    ConditionTestProcessingWriter,
    ConditionTestWriterParams,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

BIDS_ROOT = Path(r"D:\CBT\bids")

# iEEG files to analyse. These are grouped per subject.
IEEG_FILTERS = {
    "suffix": "ieeg",
    "extension": ".vhdr",
    "desc": "gammasm250",
}

# Optional secondary tables used by the task-specific resolver.
# Adjust these filters to match where your events/behaviour tables live.
SECONDARY_FILTERS = [
    {"scope": "raw", "datatype": "beh", "suffix": "beh", "extension": ".tsv"},
    {"scope": "raw", "datatype": "ieeg", "suffix": "electrodes", "extension": ".tsv"},
]

PARAMS = ConditionTestParams(
    anchor_event_codes=["10"],
    tmin_s=-2.0,
    tmax_s=2.0,
    condition_a="accepted",
    condition_b="rejected",
    #atlas_name="MarsAtlas",
    #n_bins=24,
    # Optional: remove residual line noise before epoch extraction.
    #notch_filter_freqs=[50.0],
    activity_zscore="none",
    activity_baseline_tmin_s=-0.2,
    activity_baseline_tmax_s=0.0,
    activity_baseline_scope="global",
    activity_baseline_remove_outlier_trial_means=False,
    p_value_correction_method="none",
    n_permutations=0,
    significance_alpha=0.05,
)

# This resolver is the task-specific layer for accepted vs rejected.
# Update the column names and conditions to match your dataset.
RESOLVER = TableTrialResolver(
    conditions=[
        {
            "label": "accepted",
            "when": {"column": "choice", "op": "==", "value": "1"},
        },
        {
            "label": "rejected",
            "when": {"column": "choice", "op": "==", "value": "0"},
        },
    ],
)

WRITER_PARAMS = ConditionTestWriterParams(
    bids_root=BIDS_ROOT,
    output_format="hdf5",
    output_description="simple"
)

N_JOBS = 1


if __name__ == "__main__":
    ds = BIDSDataset(BIDS_ROOT)
    groups = build_subject_groups(ds, IEEG_FILTERS, SECONDARY_FILTERS)
    print(f"Found {len(groups)} subject group(s). Running with n_jobs={N_JOBS}.")

    processor = ConditionTestProcessing(PARAMS, resolver=RESOLVER)
    writer = ConditionTestProcessingWriter(WRITER_PARAMS)

    out_paths = processor.run(groups, writer, n_jobs=N_JOBS)
    for path in out_paths:
        print(f"Wrote {path}")


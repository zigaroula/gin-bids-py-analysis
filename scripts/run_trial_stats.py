"""
Trial statistics on iEEG recordings - run script.
Edit the parameters below and run: python scripts/run_trial_stats.py
"""

from __future__ import annotations

from pathlib import Path

from gin_bids_py_analysis.bids import BIDSDataset, BIDSFileGroup, build_subject_groups
from gin_bids_py_analysis.processing.trial_stats import (
    TableTrialLabelResolver,
    TrialStatsParams,
    TrialStatsProcessing,
    TrialStatsProcessingWriter,
    TrialStatsWriterParams,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

BIDS_ROOT = Path(r"E:\CBT\bids")

# iEEG files to analyse. These are grouped per subject.
IEEG_FILTERS = {
    "suffix": "ieeg",
    "extension": ".vhdr",
    "desc": "gammasm0",
}

# Optional secondary tables used by the task-specific resolver.
# Adjust these filters to match where your events/behaviour tables live.
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
    #atlas_name="MarsAtlas",
    #n_bins=24,
    p_value_correction_method="permutation",
    n_permutations=500,
    significance_alpha=0.05,
)

# This resolver is the task-specific layer for accepted vs rejected.
# Update the column names and label map to match your dataset.
RESOLVER = TableTrialLabelResolver(
    label_column="choice",
    label_map={
        "0": "rejected",
        "1": "accepted",
    },
)

WRITER_PARAMS = TrialStatsWriterParams(
    bids_root=BIDS_ROOT,
    output_format="hdf5"
)

N_JOBS = 1


if __name__ == "__main__":
    ds = BIDSDataset(BIDS_ROOT)
    groups = build_subject_groups(ds, IEEG_FILTERS, SECONDARY_FILTERS)
    print(f"Found {len(groups)} subject group(s). Running with n_jobs={N_JOBS}.")

    processor = TrialStatsProcessing(PARAMS, resolver=RESOLVER)
    writer = TrialStatsProcessingWriter(WRITER_PARAMS)

    out_paths = processor.run(groups, writer, n_jobs=N_JOBS)
    for path in out_paths:
        print(f"Wrote {path}")

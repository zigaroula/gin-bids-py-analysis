"""
Trial slope statistics on iEEG recordings - run script.
Edit the parameters below and run: python scripts/run_trial_slope_stats.py
"""

from __future__ import annotations

from pathlib import Path

from gin_bids_py_analysis.bids import BIDSDataset, build_subject_groups
from gin_bids_py_analysis.processing.trial_slope_stats import (
    TrialSlopeStatsParams,
    TrialSlopeStatsProcessing,
    TrialSlopeStatsProcessingWriter,
    TrialSlopeStatsWriterParams,
)
from gin_bids_py_analysis.processing.trial_stats import TableTrialLabelResolver

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

BIDS_ROOT = Path(r"D:\CBT\bids")

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
    anchor_event_codes=["10"],
    tmin_s=-2.0,
    tmax_s=2.0,
    condition_a="accepted",
    condition_b="rejected",
    predictor_metadata_key="predictor_value",
    p_value_correction_method="fdr_bh",
    significance_alpha=0.05,
)

# Update column names to match your dataset.
RESOLVER = TableTrialLabelResolver(
    label_column="choice",
    label_map={
        "0": "rejected",
        "1": "accepted",
    },
    extra_metadata_columns={
        "predictor_value": "value_for_slope",
    },
)

WRITER_PARAMS = TrialSlopeStatsWriterParams(
    bids_root=BIDS_ROOT,
    output_format="hdf5",
    output_description="simple",
)

N_JOBS = 1


def main() -> list[Path]:
    ds = BIDSDataset(BIDS_ROOT)
    groups = build_subject_groups(ds, IEEG_FILTERS, SECONDARY_FILTERS)
    print(f"Found {len(groups)} subject group(s). Running with n_jobs={N_JOBS}.")

    processor = TrialSlopeStatsProcessing(PARAMS, resolver=RESOLVER)
    writer = TrialSlopeStatsProcessingWriter(WRITER_PARAMS)

    out_paths = processor.run(groups, writer, n_jobs=N_JOBS)
    for path in out_paths:
        print(f"Wrote {path}")
    return out_paths


if __name__ == "__main__":
    main()

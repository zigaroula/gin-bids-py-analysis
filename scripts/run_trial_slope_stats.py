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
from gin_bids_py_analysis.processing.utils.trial_resolver import TableTrialResolver

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

BIDS_ROOT = Path(r"E:\data_clarissa\valuation\bids")

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
    experiment_start_event_code="5",
    tmin_s=-1.0,
    tmax_s=6.0,
    condition_a="pleasant",
    condition_b="unpleasant",
    predictor="rating",
    activity_zscore="none",
    activity_baseline_tmin_s=-0.2,
    activity_baseline_tmax_s=0.0,
    p_value_correction_method="none",
    significance_alpha=0.05,
)

RESOLVER = TableTrialResolver(
    conditions=[
        {
            "label": "pleasant",
            "when": {"column": "pleasant", "op": "==", "value": "2.0"},
        },
        {
            "label": "unpleasant",
            "when": {"column": "pleasant", "op": "==", "value": "1.0"},
        },
    ],
    extract_columns=["rating"],
)

WRITER_PARAMS = TrialSlopeStatsWriterParams(
    bids_root=BIDS_ROOT,
    output_format="hdf5",
    output_description="correlation",
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



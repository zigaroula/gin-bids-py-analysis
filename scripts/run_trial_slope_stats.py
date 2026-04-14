"""
Trial slope statistics on iEEG recordings - run script.
Edit the parameters below and run: python scripts/run_trial_slope_stats.py
"""

from __future__ import annotations

from pathlib import Path

from gin_bids_py_analysis.bids import BIDSDataset, build_subject_groups
from gin_bids_py_analysis.processing.trial_stats.regression import (
    RegressionParams,
    RegressionProcessing,
    RegressionProcessingWriter,
    RegressionWriterParams,
)
from gin_bids_py_analysis.processing.utils.trial_resolver import TableTrialResolver

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

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

PARAMS = RegressionParams(
    anchor_event_codes=["11", "12"],
    experiment_start_event_code="5",
    tmin_s=-0.5,
    tmax_s=5.0,
    condition_a="pleasant",
    condition_b="unpleasant",
    predictor="rating",
    predictor_transform_by_condition={
        "pleasant": {"scale": 1.0, "offset": 0.0},
        "unpleasant": {"scale": -1.0, "offset": 0.0},
    },
    predictor_zscore="none",
    activity_zscore="baseline",
    activity_baseline_tmin_s=-0.25,
    activity_baseline_tmax_s=-0.05,
    activity_baseline_scope="global",
    activity_baseline_remove_outlier_trial_means=True,
    p_value_correction_method="none",
    significance_alpha=0.05,
    trial_activity_summary={
        "kind": "anchor_to_response_mean",
        "response": {"source": "table_column", "column": "RT", "units": "s"},
    },
    epoch_cleaning={
        "reject_trials_by_epoch_mean": True,
        "reject_trials_by_epoch_max": True,
        "reject_channels_by_trial_mean_spread": True,
        "reject_channels_by_trial_max_spread": True,
        "max_nan_trial_ratio": 0.25,
    },
    n_permutations=500
)

RESOLVER = TableTrialResolver(
    conditions=[
        {
            "label": "pleasant",
            "when": {
                "all": [
                    {"column": "pleasant", "op": "==", "value": 1},
                    {"column": "rating", "op": ">=", "value": 0},
                ]
            },
        },
        {
            "label": "unpleasant",
            "when": {
                "all": [
                    {"column": "pleasant", "op": "==", "value": 2},
                    {"column": "rating", "op": ">=", "value": 0},
                ]
            },
        },
    ],
    extract_columns=["rating", "RT"],
)

WRITER_PARAMS = RegressionWriterParams(
    bids_root=BIDS_ROOT,
    output_format="hdf5",
    output_description="correlation",
)

N_JOBS = 1


def main() -> list[Path]:
    ds = BIDSDataset(BIDS_ROOT)
    groups = build_subject_groups(ds, IEEG_FILTERS, SECONDARY_FILTERS)
    print(f"Found {len(groups)} subject group(s). Running with n_jobs={N_JOBS}.")

    processor = RegressionProcessing(PARAMS, resolver=RESOLVER)
    writer = RegressionProcessingWriter(WRITER_PARAMS)

    out_paths = processor.run(groups, writer, n_jobs=N_JOBS)
    for path in out_paths:
        print(f"Wrote {path}")
    return out_paths


if __name__ == "__main__":
    main()



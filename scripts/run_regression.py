"""Run subject-level trial-wise regression on iEEG data.

Edit the configuration block below, then run from the repository root:

    python scripts/run_regression.py
"""

from __future__ import annotations

from pathlib import Path

from bidsforge.bids import BIDSDataset, build_subject_groups
from bidsforge.processing.trial_stats import TableTrialResolver
from bidsforge.processing.trial_stats.regression import (
    RegressionParams,
    RegressionProcessing,
    RegressionProcessingWriter,
    RegressionWriterParams,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BIDS_ROOT = Path("path/to/bids_dataset")

# Point this at the data to epoch. The default assumes the Hilbert script has
# already produced derivatives/ hilbert files with desc-hilbert.
IEEG_FILTERS = {
    "scope": "hilbert",
    "datatype": "ieeg",
    "suffix": "ieeg",
    "extension": ".h5",
    "desc": "hilbert",
}

SECONDARY_FILTERS = [
    {"scope": "raw", "datatype": "beh", "suffix": "beh", "extension": ".tsv"},
    {"scope": "raw", "datatype": "ieeg", "suffix": "events", "extension": ".tsv"},
    {"scope": "raw", "datatype": "ieeg", "suffix": "electrodes", "extension": ".tsv"},
]

N_JOBS = 1
SKIP_EXISTING = True

PARAMS = RegressionParams(
    anchor_event_codes=["stimulus"],
    tmin_s=-0.5,
    tmax_s=1.5,
    condition_a="condition_a",
    condition_b="condition_b",
    min_trials_per_condition=3,
    predictor="predictor_value",
    predictor_zscore="condition",
    hilbert_smoothing_window_ms=250,
    activity_zscore="baseline",
    activity_baseline_tmin_s=-0.3,
    activity_baseline_tmax_s=-0.05,
    activity_baseline_scope="global",
    p_value_correction_method="fdr_bh",
    significance_alpha=0.05,
    n_permutations=0,
    permutation_seed=1,
)

# Regression uses the same trial resolver idea as condition_test, with one
# extra requirement: extract_columns must include the numeric predictor column
# named by PARAMS.predictor. Each condition gets its own slope map.
RESOLVER = TableTrialResolver(
    conditions=[
        {"label": "condition_a", "when": {"column": "condition", "op": "==", "value": "A"}},
        {"label": "condition_b", "when": {"column": "condition", "op": "==", "value": "B"}},
    ],
    extract_columns=["condition", "predictor_value"],
    trial_id_column="trial_id",
    anchor_onset_column="onset",
    anchor_event_code_column="anchor_event_code",
    keep_column="keep",
    filter={"suffix": "beh"},
)

# Writer parameters control where the derivative is written under
# derivatives/regression and whether individual epochs are kept in the output.
WRITER_PARAMS = RegressionWriterParams(
    bids_root=BIDS_ROOT,
    output_format="hdf5",
    output_description="regression",
    include_epochs=False,
)


def main() -> list[Path]:
    dataset = BIDSDataset(BIDS_ROOT)
    # Each subject group contains one or more iEEG files plus matching behaviour
    # and electrode tables discovered with SECONDARY_FILTERS.
    groups = build_subject_groups(dataset, IEEG_FILTERS, SECONDARY_FILTERS)
    print(f"Found {len(groups)} subject group(s). Running with n_jobs={N_JOBS}.")

    processor = RegressionProcessing(PARAMS, resolver=RESOLVER)
    writer = RegressionProcessingWriter(WRITER_PARAMS)
    out_paths = processor.run(groups, writer, n_jobs=N_JOBS, skip_existing=SKIP_EXISTING)

    for path in out_paths:
        print(f"Wrote {path}")
    return out_paths


if __name__ == "__main__":
    main()

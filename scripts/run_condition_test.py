"""Run subject-level condition-test statistics on iEEG data.

Edit the configuration block below, then run from the repository root:

    python scripts/run_condition_test.py
"""

from __future__ import annotations

from pathlib import Path

from bidsforge.bids import BIDSDataset, build_subject_groups
from bidsforge.processing.trial_stats import TableTrialResolver
from bidsforge.processing.trial_stats.condition_test import (
    ConditionTestParams,
    ConditionTestProcessing,
    ConditionTestProcessingWriter,
    ConditionTestWriterParams,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BIDS_ROOT = Path("path/to/bids_dataset")

# Point this at the iEEG data to epoch. This can be raw data or a Hilbert
# derivative written by scripts/run_hilbert.py.
IEEG_FILTERS = {
    "scope": "hilbert",
    "datatype": "ieeg",
    "suffix": "ieeg",
    "extension": ".h5",
    "desc": "hilbert",
}

# Trial tables are generic TSV/CSV files carried as secondaries. The resolver
# below expects columns such as trial_id, onset, anchor_event_code, condition,
# and keep; adapt the names and filters to your dataset.
SECONDARY_FILTERS = [
    {"scope": "raw", "datatype": "beh", "suffix": "beh", "extension": ".tsv"},
    {"scope": "raw", "datatype": "ieeg", "suffix": "events", "extension": ".tsv"},
    {"scope": "raw", "datatype": "ieeg", "suffix": "electrodes", "extension": ".tsv"},
]

N_JOBS = 1
SKIP_EXISTING = True

PARAMS = ConditionTestParams(
    anchor_event_codes=["stimulus"],
    tmin_s=-0.5,
    tmax_s=1.5,
    condition_a="condition_a",
    condition_b="condition_b",
    min_trials_per_condition=2,
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

# TableTrialResolver turns rows from a TSV/CSV trial table into trial labels.
# Here, rows with condition == "A" become condition_a and rows with
# condition == "B" become condition_b. Edit the column names and rules to
# match your own behaviour/events table.
RESOLVER = TableTrialResolver(
    conditions=[
        {"label": "condition_a", "when": {"column": "condition", "op": "==", "value": "A"}},
        {"label": "condition_b", "when": {"column": "condition", "op": "==", "value": "B"}},
    ],
    extract_columns=["condition"],
    trial_id_column="trial_id",
    anchor_onset_column="onset",
    anchor_event_code_column="anchor_event_code",
    keep_column="keep",
    filter={"suffix": "beh"},
)

# Writer parameters control where the derivative is written under
# derivatives/condition_test and how much data is stored in the result file.
WRITER_PARAMS = ConditionTestWriterParams(
    bids_root=BIDS_ROOT,
    output_format="hdf5",
    output_description="conditiontest",
    include_epochs=False,
)


def main() -> list[Path]:
    dataset = BIDSDataset(BIDS_ROOT)
    # build_subject_groups pools primary iEEG files and matching secondary
    # tables by subject, so each processor call sees the signal plus metadata.
    groups = build_subject_groups(dataset, IEEG_FILTERS, SECONDARY_FILTERS)
    print(f"Found {len(groups)} subject group(s). Running with n_jobs={N_JOBS}.")

    processor = ConditionTestProcessing(PARAMS, resolver=RESOLVER)
    writer = ConditionTestProcessingWriter(WRITER_PARAMS)
    out_paths = processor.run(groups, writer, n_jobs=N_JOBS, skip_existing=SKIP_EXISTING)

    for path in out_paths:
        print(f"Wrote {path}")
    return out_paths


if __name__ == "__main__":
    main()

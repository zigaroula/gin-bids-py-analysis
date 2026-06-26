"""Run subject-level trial-wise regression on time-frequency derivatives.

Edit the configuration block below, then run from the repository root:

    python scripts/run_time_frequency_regression.py
"""

from __future__ import annotations

from pathlib import Path

from bidsforge.bids import BIDSDataset, build_subject_groups
from bidsforge.processing.trial_stats import TableTrialResolver
from bidsforge.processing.time_frequency_stats.regression import (
    TimeFrequencyRegressionParams,
    TimeFrequencyRegressionProcessing,
    TimeFrequencyRegressionWriter,
    TimeFrequencyRegressionWriterParams,
)


BIDS_ROOT = Path("path/to/bids_dataset")

TFR_FILTERS = {
    "scope": "time_frequency",
    "datatype": "ieeg",
    "suffix": "tfr",
    "extension": ".h5",
}

SECONDARY_FILTERS = [
    {"scope": "raw", "datatype": "beh", "suffix": "beh", "extension": ".tsv"},
]

N_JOBS = 1
SKIP_EXISTING = True

PARAMS = TimeFrequencyRegressionParams(
    anchor_event_codes=[],
    time_window_s=(-1.0, 3.5),
    time_selection="strict",
    power_mode="stored",
    condition_a="pleasant",
    condition_b="unpleasant",
    predictor="rating_z",
    predictor_zscore="none",
    p_value_correction_method="none",
)

RESOLVER = TableTrialResolver(
    conditions=[
        {"label": "pleasant", "when": {"column": "pleasantness", "op": "==", "value": "1"}},
        {"label": "unpleasant", "when": {"column": "pleasantness", "op": "==", "value": "2"}},
    ],
    extract_columns=["pleasantness", "rating_z"],
    filter={"suffix": "beh"},
)

WRITER_PARAMS = TimeFrequencyRegressionWriterParams(
    bids_root=BIDS_ROOT,
    output_format="hdf5",
    output_description="tfregression",
    include_epochs=False,
)


def main() -> list[Path]:
    dataset = BIDSDataset(BIDS_ROOT)
    groups = build_subject_groups(dataset, TFR_FILTERS, SECONDARY_FILTERS)
    print(f"Found {len(groups)} subject group(s). Running with n_jobs={N_JOBS}.")
    processor = TimeFrequencyRegressionProcessing(PARAMS, resolver=RESOLVER)
    writer = TimeFrequencyRegressionWriter(WRITER_PARAMS)
    out_paths = processor.run(groups, writer, n_jobs=N_JOBS, skip_existing=SKIP_EXISTING)
    for path in out_paths:
        print(f"Wrote {path}")
    return out_paths


if __name__ == "__main__":
    main()


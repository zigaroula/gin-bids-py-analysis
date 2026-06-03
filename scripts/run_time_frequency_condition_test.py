"""Run subject-level condition statistics on time-frequency derivatives.

Edit the configuration block below, then run from the repository root:

    python scripts/run_time_frequency_condition_test.py
"""

from __future__ import annotations

from pathlib import Path

from bidsforge.bids import BIDSDataset, build_subject_groups
from bidsforge.processing.trial_stats import TableTrialResolver
from bidsforge.processing.time_frequency_stats.condition_test import (
    TimeFrequencyConditionTestParams,
    TimeFrequencyConditionTestProcessing,
    TimeFrequencyConditionTestWriter,
    TimeFrequencyConditionTestWriterParams,
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

PARAMS = TimeFrequencyConditionTestParams(
    anchor_event_codes=[],
    time_window_s=(-1.5, 2.0),
    time_selection="strict_matlab",
    power_mode="stored",
    condition_a="switch_hit",
    condition_b="nonswitch_hit",
    equal_var=False,
    compute_grand_average=False,
    p_value_correction_method="none",
)

RESOLVER = TableTrialResolver(
    conditions=[
        {
            "label": "switch_hit",
            "when": {
                "all": [
                    {"column": "condition", "op": "==", "value": "sw"},
                    {"column": "response", "op": "==", "value": "1"},
                ],
            },
        },
        {
            "label": "nonswitch_hit",
            "when": {
                "all": [
                    {"column": "condition", "op": "!=", "value": "sw"},
                    {"column": "response", "op": "==", "value": "1"},
                ],
            },
        },
    ],
    extract_columns=["condition", "response"],
    filter={"suffix": "beh"},
)

WRITER_PARAMS = TimeFrequencyConditionTestWriterParams(
    bids_root=BIDS_ROOT,
    output_format="hdf5",
    output_description="tfconditiontest",
    include_epochs=False,
)


def main() -> list[Path]:
    dataset = BIDSDataset(BIDS_ROOT)
    groups = build_subject_groups(dataset, TFR_FILTERS, SECONDARY_FILTERS)
    print(f"Found {len(groups)} subject group(s). Running with n_jobs={N_JOBS}.")
    processor = TimeFrequencyConditionTestProcessing(PARAMS, resolver=RESOLVER)
    writer = TimeFrequencyConditionTestWriter(WRITER_PARAMS)
    out_paths = processor.run(groups, writer, n_jobs=N_JOBS, skip_existing=SKIP_EXISTING)
    for path in out_paths:
        print(f"Wrote {path}")
    return out_paths


if __name__ == "__main__":
    main()


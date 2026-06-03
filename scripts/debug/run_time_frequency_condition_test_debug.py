#!/usr/bin/env python3
"""Generate Python TF condition-test HDF5 outputs for Clarissa debug data."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from bidsforge.bids import BIDSDataset
from bidsforge.processing.time_frequency_stats.condition_test import (
    TimeFrequencyConditionTestProcessing,
)

from time_frequency_stats_debug_shared import (  # noqa: E402
    BIDS_ROOT,
    TrialSlopeConditionResolver,
    build_condition_test_params,
    build_condition_test_writer,
    build_tfr_stats_groups,
)
from trial_slope_shared import TRIAL_SLOPE_N_JOBS, print_recipe_summary  # noqa: E402


SKIP_EXISTING = False


def main() -> list[Path]:
    print_recipe_summary()
    dataset = BIDSDataset(BIDS_ROOT)
    groups = build_tfr_stats_groups(dataset)
    print(f"Found {len(groups)} TFR group(s). Running with n_jobs={TRIAL_SLOPE_N_JOBS}.")

    processor = TimeFrequencyConditionTestProcessing(
        build_condition_test_params(),
        resolver=TrialSlopeConditionResolver(),
    )
    writer = build_condition_test_writer()
    out_paths = processor.run(
        groups,
        writer,
        n_jobs=TRIAL_SLOPE_N_JOBS,
        skip_existing=SKIP_EXISTING,
    )
    for path in out_paths:
        print(f"Wrote {path}")
    return out_paths


if __name__ == "__main__":
    main()

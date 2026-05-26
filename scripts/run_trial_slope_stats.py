"""
Trial slope statistics on iEEG recordings - run script.

Edit the shared recipe in scripts/trial_slope_shared.py, then run:

    python scripts/run_trial_slope_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from gin_bids_py_analysis.bids import BIDSDataset
from gin_bids_py_analysis.processing.trial_stats.regression import (
    RegressionProcessing,
    RegressionProcessingWriter,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from trial_slope_shared import (  # noqa: E402
    BIDS_ROOT,
    RESOLVER,
    TRIAL_SLOPE_N_JOBS,
    TRIAL_SLOPE_SKIP_EXISTING,
    build_params,
    build_regression_writer_params,
    build_trial_annotators,
    build_trial_slope_groups,
    load_roi_channels_from_csv,
    print_recipe_summary,
    print_roi_summary,
)


def main() -> list[Path]:
    print_recipe_summary()
    manual_region_channels = load_roi_channels_from_csv()
    print_roi_summary(manual_region_channels)
    annotators = build_trial_annotators(manual_region_channels)

    ds = BIDSDataset(BIDS_ROOT)
    groups = build_trial_slope_groups(ds)
    print(
        f"Found {len(groups)} subject group(s). "
        f"Running with n_jobs={TRIAL_SLOPE_N_JOBS}."
    )

    writer = RegressionProcessingWriter(build_regression_writer_params())
    out_paths: list[Path] = []
    for group in groups:
        primary = getattr(group, "primary", None)
        subject_id = primary.get("subject") if primary is not None else ""
        subject_id = subject_id or ""
        params = build_params(subject_id)
        processor = RegressionProcessing(params, resolver=RESOLVER, annotators=annotators)
        run_kwargs = {"n_jobs": TRIAL_SLOPE_N_JOBS}
        if TRIAL_SLOPE_SKIP_EXISTING:
            run_kwargs["skip_existing"] = True
        paths = processor.run([group], writer, **run_kwargs)
        out_paths.extend(paths)
    for path in out_paths:
        print(f"Wrote {path}")
    return out_paths


if __name__ == "__main__":
    main()


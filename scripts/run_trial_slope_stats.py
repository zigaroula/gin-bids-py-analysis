"""
Trial slope statistics on iEEG recordings - run script.
Edit the parameters below and run: python scripts/run_trial_slope_stats.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from gin_bids_py_analysis.bids import BIDSDataset
from gin_bids_py_analysis.processing.trial_stats.regression import (
    RegressionProcessing,
    RegressionProcessingWriter,
    RegressionWriterParams,
)

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from trial_slope_shared import (  # noqa: E402
    BIDS_ROOT,
    RESOLVER,
    build_params,
    build_trial_annotators,
    build_trial_slope_groups,
    load_roi_channels_from_csv,
    print_roi_summary,
)

WRITER_PARAMS = RegressionWriterParams(
    bids_root=BIDS_ROOT,
    output_format="hdf5",
    output_description="onsetnospike",
)

N_JOBS = 1


def main() -> list[Path]:
    manual_region_channels = load_roi_channels_from_csv()
    print_roi_summary(manual_region_channels)
    annotators = build_trial_annotators(manual_region_channels)

    ds = BIDSDataset(BIDS_ROOT)
    groups = build_trial_slope_groups(ds)
    print(f"Found {len(groups)} subject group(s). Running with n_jobs={N_JOBS}.")

    writer = RegressionProcessingWriter(WRITER_PARAMS)
    out_paths: list[Path] = []
    for group in groups:
        subject_id = group.primary.get("subject") or ""
        params = build_params(subject_id)
        processor = RegressionProcessing(params, resolver=RESOLVER, annotators=annotators)
        paths = processor.run([group], writer, n_jobs=N_JOBS, skip_existing=False)
        out_paths.extend(paths)
    for path in out_paths:
        print(f"Wrote {path}")
    return out_paths


if __name__ == "__main__":
    main()

"""
Time-frequency b1 run script.

Edit the shared recipe in scripts/debug/trial_slope_shared.py, then run:

    python scripts/debug/run_time_frequency.py
"""

from __future__ import annotations

from collections import defaultdict
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
for path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bidsforge.bids import BIDSDataset
from bidsforge.processing.time_frequency import (
    TimeFrequencyProcessing,
    TimeFrequencyProcessingWriter,
)

from trial_slope_shared import (  # noqa: E402
    BIDS_ROOT,
    TIME_FREQUENCY_N_JOBS,
    TIME_FREQUENCY_SKIP_EXISTING,
    build_time_frequency_groups,
    build_time_frequency_params,
    build_time_frequency_writer_params,
    print_recipe_summary,
)


def main() -> list[Path]:
    print_recipe_summary()
    ds = BIDSDataset(BIDS_ROOT)
    groups = build_time_frequency_groups(ds)
    print(
        f"Found {len(groups)} file group(s). "
        f"Running with n_jobs={TIME_FREQUENCY_N_JOBS}."
    )

    writer = TimeFrequencyProcessingWriter(build_time_frequency_writer_params())

    groups_by_shift = defaultdict(list)
    params_by_shift = {}
    for group in groups:
        subject_id = group.primary.get("subject") or ""
        params = build_time_frequency_params(subject_id)
        shift = int(params.event_sample_shift_samples)
        groups_by_shift[shift].append(group)
        params_by_shift[shift] = params

    out_paths = []
    for shift, shifted_groups in sorted(groups_by_shift.items()):
        print(
            f"Running {len(shifted_groups)} group(s) with "
            f"event_sample_shift_samples={shift}."
        )
        processor = TimeFrequencyProcessing(params_by_shift[shift])
        out_paths.extend(
            processor.run(
                shifted_groups,
                writer,
                n_jobs=TIME_FREQUENCY_N_JOBS,
                skip_existing=TIME_FREQUENCY_SKIP_EXISTING,
            )
        )

    for path in out_paths:
        print(f"Wrote {path}")
    return out_paths


if __name__ == "__main__":
    main()

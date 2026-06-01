"""
Hilbert analysis run script.

Edit the shared recipe in scripts/debug/trial_slope_shared.py, then run:

    python scripts/debug/run_hilbert.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
for path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bidsforge.bids import BIDSDataset
from bidsforge.processing.hilbert import (
    HilbertProcessing,
    HilbertProcessingWriter,
)

from trial_slope_shared import (  # noqa: E402
    BIDS_ROOT,
    HILBERT_N_JOBS,
    HILBERT_SKIP_EXISTING,
    build_hilbert_groups,
    build_hilbert_params,
    build_hilbert_writer_params,
    print_recipe_summary,
)


def main() -> list[Path]:
    print_recipe_summary()
    ds = BIDSDataset(BIDS_ROOT)
    groups = build_hilbert_groups(ds)
    print(f"Found {len(groups)} file group(s). Running with n_jobs={HILBERT_N_JOBS}.")

    processor = HilbertProcessing(build_hilbert_params())
    writer = HilbertProcessingWriter(build_hilbert_writer_params())

    out_paths = processor.run(
        groups,
        writer,
        n_jobs=HILBERT_N_JOBS,
        skip_existing=HILBERT_SKIP_EXISTING,
    )
    for path in out_paths:
        print(f"Wrote {path}")
    return out_paths


if __name__ == "__main__":
    main()


"""
Hilbert analysis — run script.
Edit the parameters below and run: python scripts/run_hilbert.py
"""

from pathlib import Path

from gin_bids_py_analysis.bids import BIDSDataset
from gin_bids_py_analysis.processing.hilbert import (
    HilbertParams,
    HilbertProcessing,
    HilbertProcessingWriter,
    HilbertWriterParams,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

BIDS_ROOT = Path("/data/my_study")

# BIDS entity filters: only files matching ALL of these will be processed.
# Remove any key you don't want to filter on.
FILE_FILTERS = {
    "suffix": "ieeg",
    "extension": ".vhdr",
    # "subject": "01",
    # "session": "01",
    # "task": "rest",
}

PARAMS = HilbertParams(
    freq_bands=[(1, 4), (8, 12), (30, 80)],
    sfreq=1000.0,
)

WRITER_PARAMS = HilbertWriterParams(
    bids_root=BIDS_ROOT
)

N_JOBS = 1  # parallelism across files; set to -1 to use all available CPUs

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = BIDSDataset(BIDS_ROOT)
    files = ds.get_files(**FILE_FILTERS)
    print(f"Found {len(files)} file(s). Running with n_jobs={N_JOBS}.")

    processor = HilbertProcessing(PARAMS)
    writer = HilbertProcessingWriter(WRITER_PARAMS)

    out_paths = processor.run(files, writer, n_jobs=N_JOBS)
    for p in out_paths:
        print(f"Wrote {p}")

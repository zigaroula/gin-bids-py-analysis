"""
Delphos analysis — run script.
Edit the parameters below and run: python scripts/run_delphos.py
"""

from pathlib import Path

from gin_bids_py_analysis.bids import BIDSDataset
from gin_bids_py_analysis.processing.delphos import (
    DelphosParams,
    DelphosProcessing,
    DelphosProcessingWriter,
    DelphosWriterParams,
)
from gin_bids_py_analysis.processing.utils.channels import (
    BipolarDirection,
    BipolarStorage,
    MontageMode,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

BIDS_ROOT = Path(r"D:\CBT\bids")

# BIDS entity filters: only files matching ALL of these will be processed.
# Remove any key you don't want to filter on.
FILE_FILTERS = {
    "suffix": "ieeg",
    "extension": ".vhdr",
    #"subject": "epi01",
    #"run": "01",
}

# Algorithm parameters for detection
PARAMS = DelphosParams(
    channels_for_montage=r'[A-Z]p?([0-1][0-9])'
)

# Writer configuration for output files
WRITER_PARAMS = DelphosWriterParams(
    bids_root=BIDS_ROOT,
    output_format="tsv"
)

N_JOBS = 1  # parallelism across files; set to -1 to use all available CPUs

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = BIDSDataset(BIDS_ROOT)
    files = ds.get_files(pipeline="raw", **FILE_FILTERS)
    print(f"Found {len(files)} file(s). Running with n_jobs={N_JOBS}.")

    processor = DelphosProcessing(PARAMS)
    writer = DelphosProcessingWriter(WRITER_PARAMS)

    out_paths = processor.run(files, writer, n_jobs=N_JOBS)
    for p in out_paths:
        print(f"Wrote {p}")

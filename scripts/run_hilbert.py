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

# Frequency grid: bins [f_min, f_min+f_step, ..., f_max]
# Adjacent pairs form subbands: (50-60 Hz), (60-70 Hz), ..., (140-150 Hz)
PARAMS = HilbertParams(
    f_min=50,
    f_max=150,
    f_step=10,
    downsampled_frequency_hz=64.0,
    montage_mode=MontageMode.BIPOLAR,
    bipolar_direction=BipolarDirection.NEXT_MINUS_PREVIOUS,
    bipolar_storage=BipolarStorage.NEXT,
    smoothing_windows_ms=[0, 250, 500, 1000, 2500, 5000],
    do_downsample=True,
    do_normalize_percent=True,
    do_smoothing=True
)

WRITER_PARAMS = HilbertWriterParams(
    bids_root=BIDS_ROOT,
    output_description="gamma",
    output_extension=".vhdr"
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

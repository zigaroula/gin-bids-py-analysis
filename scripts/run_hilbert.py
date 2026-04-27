"""
Hilbert analysis — run script.
Edit the parameters below and run: python scripts/run_hilbert.py
"""

from pathlib import Path

from gin_bids_py_analysis.bids import BIDSDataset, build_subject_groups
from gin_bids_py_analysis.processing.hilbert import (
    HilbertParams,
    HilbertProcessing,
    HilbertProcessingWriter,
    HilbertWriterParams,
    NormalizationMode,
    ProcessingMethod,
)
from gin_bids_py_analysis.processing.utils.channels import (
    BipolarDirection,
    BipolarStorage,
    MontageMode,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

BIDS_ROOT = Path(r"D:\Boulot\clarissa_bids")

# BIDS entity filters: only files matching ALL of these will be processed.
# Remove any key you don't want to filter on.
FILE_FILTERS = {
    "suffix": "ieeg",
    "extension": ".vhdr",
    "subject": "PRA2021AAAb"
    #"run": "01",
}

# Frequency grid: bins [f_min, f_min+f_step, ..., f_max]
# Adjacent pairs form subbands: (50-60 Hz), (60-70 Hz), ..., (140-150 Hz)
PARAMS = HilbertParams(
    f_min=50,
    f_max=150,
    f_step=10,
    method=ProcessingMethod.SPM2ENV,
    computation_frequency_hz=512.0,
    downsampled_frequency_hz=100.0,
    # SPM2ENV BrainVision export projects events with SPM's continuous-file
    # sample convention, so no extra source-sample shift is needed here.
    event_sample_shift_samples=0,
    events_source="events_tsv",
    smoothing_windows_ms= [0, 250, 500, 1000, 2500, 5000],
    montage_mode=MontageMode.BIPOLAR,
    bipolar_direction=BipolarDirection.NEXT_MINUS_PREVIOUS,
    bipolar_storage=BipolarStorage.NEXT,
    normalization_mode=NormalizationMode.PERCENT,
    channels_to_exclude_for_montage=r'(?:MKR|DELD|DELG|EOG|ECG|EMG|DC|EXG|EKG|REF|GND|EMPTY).*',
    #channels_for_montage=r'[A-Z]p?([0-1][0-9])'
)

WRITER_PARAMS = HilbertWriterParams(
    bids_root=BIDS_ROOT,
    output_description="bga",
    output_format="brainvision"
)

SECONDARY_FILTERS = [
    {
        "scope": "raw",
        "suffix": "events",
        "extension": ".tsv",
        "datatype": "ieeg",
    },
]

N_JOBS = 1  # parallelism across files; set to -1 to use all available CPUs

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ds = BIDSDataset(BIDS_ROOT)
    groups = build_subject_groups(
        ds,
        {"scope": "raw", **FILE_FILTERS},
        SECONDARY_FILTERS,
        aggregate_runs=False,
    )
    print(f"Found {len(groups)} file group(s). Running with n_jobs={N_JOBS}.")

    processor = HilbertProcessing(PARAMS)
    writer = HilbertProcessingWriter(WRITER_PARAMS)

    out_paths = processor.run(groups, writer, n_jobs=N_JOBS)
    for p in out_paths:
        print(f"Wrote {p}")

"""Run the Hilbert envelope pipeline on a BIDS iEEG dataset.

Edit the configuration block below, then run from the repository root:

    python scripts/run_hilbert.py
"""

from __future__ import annotations

from pathlib import Path

from bidsforge.bids import BIDSDataset, BIDSFileGroup
from bidsforge.processing.hilbert import (
    HilbertParams,
    HilbertProcessing,
    HilbertProcessingWriter,
    HilbertWriterParams,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BIDS_ROOT = Path("path/to/bids_dataset")

# Select the recordings to process. Add entities such as "subject", "session",
# "task", or "run" to narrow the selection.
IEEG_FILTERS = {
    "scope": "raw",
    "datatype": "ieeg",
    "suffix": "ieeg",
    "extension": ".vhdr",
}

# Optional matching *_events.tsv files. They are used only when EVENTS_SOURCE is
# "events_tsv" or "auto".
EVENTS_FILTERS = {
    "scope": "raw",
    "datatype": "ieeg",
    "suffix": "events",
    "extension": ".tsv",
}

EVENTS_SOURCE = "auto"  # "annotations", "events_tsv", or "auto"
N_JOBS = 1
SKIP_EXISTING = True

PARAMS = HilbertParams(
    f_min=50.0,
    f_max=150.0,
    f_step=10.0,
    computation_frequency_hz=None,
    downsampled_frequency_hz=64.0,
    events_source=EVENTS_SOURCE,
    notch_filter_freqs=[],  # Example: [50.0, 100.0]
    montage_mode="mono",  # "mono" or "bipolar"
    smoothing_windows_ms=[0, 250],
    normalization_mode="percent",
    method="localizer",
)

WRITER_PARAMS = HilbertWriterParams(
    bids_root=BIDS_ROOT,
    output_format="hdf5",  # "hdf5", "brainvision", or "matlab"
    output_description="hilbert",
)


def build_groups(dataset: BIDSDataset) -> list[BIDSFileGroup]:
    ieeg_files = sorted(dataset.get_files(**IEEG_FILTERS), key=lambda file: str(file.path))
    event_files = sorted(dataset.get_files(**EVENTS_FILTERS), key=lambda file: str(file.path))

    groups: list[BIDSFileGroup] = []
    for ieeg_file in ieeg_files:
        subject = ieeg_file.get("subject")
        session = ieeg_file.get("session")
        run = ieeg_file.get("run")
        task = ieeg_file.get("task")
        secondaries = [
            file
            for file in event_files
            if file.get("subject") == subject
            and (session is None or file.get("session") in (None, session))
            and (run is None or file.get("run") in (None, run))
            and (task is None or file.get("task") in (None, task))
        ]
        groups.append(BIDSFileGroup(primary=ieeg_file, secondaries=secondaries))
    return groups


def main() -> list[Path]:
    dataset = BIDSDataset(BIDS_ROOT)
    groups = build_groups(dataset)
    print(f"Found {len(groups)} iEEG recording(s). Running with n_jobs={N_JOBS}.")

    processor = HilbertProcessing(PARAMS)
    writer = HilbertProcessingWriter(WRITER_PARAMS)
    out_paths = processor.run(groups, writer, n_jobs=N_JOBS, skip_existing=SKIP_EXISTING)

    for path in out_paths:
        print(f"Wrote {path}")
    return out_paths


if __name__ == "__main__":
    main()

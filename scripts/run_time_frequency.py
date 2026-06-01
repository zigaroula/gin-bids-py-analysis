"""Run the time-frequency power pipeline on a BIDS iEEG dataset.

Edit the configuration block below, then run from the repository root:

    python scripts/run_time_frequency.py
"""

from __future__ import annotations

from pathlib import Path

from bidsforge.bids import BIDSDataset, BIDSFileGroup
from bidsforge.processing.time_frequency import (
    TimeFrequencyParams,
    TimeFrequencyProcessing,
    TimeFrequencyProcessingWriter,
    TimeFrequencyWriterParams,
)


BIDS_ROOT = Path("path/to/bids_dataset")

IEEG_FILTERS = {
    "scope": "raw",
    "datatype": "ieeg",
    "suffix": "ieeg",
    "extension": ".vhdr",
}

EVENTS_FILTERS = {
    "scope": "raw",
    "datatype": "ieeg",
    "suffix": "events",
    "extension": ".tsv",
}

N_JOBS = 1
SKIP_EXISTING = True

PARAMS = TimeFrequencyParams(
    anchor_event_codes=["11"],
    tmin_s=-1.5,
    tmax_s=2.0,
    events_source="auto",
    time_decimation=20,
    baseline_window_s=(-1.3, -0.7),
    apply_baseline=True,
    montage_mode="mono",
)

WRITER_PARAMS = TimeFrequencyWriterParams(
    bids_root=BIDS_ROOT,
    output_format="hdf5",
    output_description="timefrequency",
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

    processor = TimeFrequencyProcessing(PARAMS)
    writer = TimeFrequencyProcessingWriter(WRITER_PARAMS)
    out_paths = processor.run(groups, writer, n_jobs=N_JOBS, skip_existing=SKIP_EXISTING)

    for path in out_paths:
        print(f"Wrote {path}")
    return out_paths


if __name__ == "__main__":
    main()

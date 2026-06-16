"""Reprocess CBT Hilbert derivatives for subjects used in ROI plots.

Run from the repository root:

    python scripts/debug/reprocess_cbt_roi_hilbert.py

The target subject list is derived from MANUAL_REGION_CHANNELS in
run_onset_description_condition_tests.py. Outputs are BrainVision files with
desc-gammasm250 under derivatives/hilbert.
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[1]
for path in (_REPO_ROOT, _SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from bidsforge.bids import BIDSDataset, BIDSFileGroup, build_subject_groups
from bidsforge.processing.hilbert import (
    HilbertParams,
    HilbertProcessing,
    HilbertProcessingWriter,
    HilbertWriterParams,
)
from bidsforge.processing.hilbert.params import (
    NormalizationMode,
    ProcessingMethod,
)
from bidsforge.processing.utils.channels import (
    BipolarDirection,
    BipolarStorage,
    MontageMode,
)

from run_onset_description_condition_tests import MANUAL_REGION_CHANNELS


BIDS_ROOT = Path(r"D:\CBT\bids")

IEEG_FILTERS = {
    "scope": "raw",
    "datatype": "ieeg",
    "suffix": "ieeg",
    "extension": ".vhdr",
    "task": "CBT",
}

EVENTS_FILTERS = [
    {
        "scope": "raw",
        "datatype": "ieeg",
        "suffix": "events",
        "extension": ".tsv",
        "task": "CBT",
    },
]

N_JOBS = 1
SKIP_EXISTING = False
DRY_RUN = False

# Set to e.g. ["epi01", "epi03"] for a smaller temporary rerun.
SUBJECTS_OVERRIDE: list[str] | None = None

PARAMS = HilbertParams(
    f_min=50.0,
    f_max=150.0,
    f_step=10.0,
    computation_frequency_hz=512.0,
    downsampled_frequency_hz=100.0,
    event_sample_shift_samples=0,
    events_source="events_tsv",
    notch_filter_freqs=[],
    montage_mode=MontageMode.BIPOLAR,
    bipolar_direction=BipolarDirection.NEXT_MINUS_PREVIOUS,
    bipolar_storage=BipolarStorage.NEXT,
    channels_for_montage=None,
    channels_to_exclude_for_montage=(
        r"(?:MKR|DELD|DELG|EOG|ECG|EMG|DC|EXG|EKG|REF|GND|EMPTY).*"
    ),
    smoothing_windows_ms=[250],
    normalization_mode=NormalizationMode.PERCENT,
    method=ProcessingMethod.SPM2ENV,
)

WRITER_PARAMS = HilbertWriterParams(
    bids_root=BIDS_ROOT,
    output_format="brainvision",
    output_description="gamma",
)


def main() -> list[Path]:
    subjects = _target_subjects()
    print(f"Target subjects ({len(subjects)}): {', '.join(subjects)}")

    dataset = BIDSDataset(BIDS_ROOT)
    groups = _target_groups(dataset, subjects)
    found_subjects = {str(group.primary.get("subject")) for group in groups}
    missing_subjects = sorted(set(subjects) - found_subjects)

    print(f"Found {len(groups)} CBT iEEG run(s) to process.")
    if missing_subjects:
        print(f"Missing raw CBT iEEG for: {', '.join(missing_subjects)}")

    for group in groups:
        subject = group.primary.get("subject")
        run = group.primary.get("run")
        print(f"  sub-{subject} run-{run}: {group.primary.path.name}")

    if DRY_RUN:
        print("DRY_RUN=True, no Hilbert processing launched.")
        return []

    processor = HilbertProcessing(PARAMS)
    writer = HilbertProcessingWriter(WRITER_PARAMS)
    out_paths = processor.run(
        groups,
        writer,
        n_jobs=N_JOBS,
        skip_existing=SKIP_EXISTING,
    )

    for path in out_paths:
        print(f"Wrote {path}")
    return out_paths


def _target_subjects() -> list[str]:
    if SUBJECTS_OVERRIDE is not None:
        return sorted({str(subject) for subject in SUBJECTS_OVERRIDE})
    return sorted(
        {
            str(subject)
            for subjects_by_roi in MANUAL_REGION_CHANNELS.values()
            for subject in subjects_by_roi
        }
    )


def _target_groups(dataset: BIDSDataset, subjects: list[str]) -> list[BIDSFileGroup]:
    subject_set = set(subjects)
    groups = build_subject_groups(
        dataset,
        IEEG_FILTERS,
        EVENTS_FILTERS,
        aggregate_runs=False,
    )
    return [
        group
        for group in groups
        if str(group.primary.get("subject")) in subject_set
    ]


if __name__ == "__main__":
    main()

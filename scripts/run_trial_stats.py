"""
Trial statistics on iEEG recordings - run script.
Edit the parameters below and run: python scripts/run_trial_stats.py
"""

from __future__ import annotations

from pathlib import Path

from gin_bids_py_analysis.bids import BIDSDataset, BIDSFile, BIDSFileGroup
from gin_bids_py_analysis.processing.trial_stats import (
    TableTrialLabelResolver,
    TrialStatsParams,
    TrialStatsProcessing,
    TrialStatsProcessingWriter,
    TrialStatsWriterParams,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

BIDS_ROOT = Path(r"E:\CBT\bids")

# iEEG files to analyse. These are grouped per subject.
IEEG_FILTERS = {
    "suffix": "ieeg",
    "extension": ".vhdr",
    "desc": "gammasm0",
}

# Optional secondary tables used by the task-specific resolver.
# Adjust these filters to match where your events/behaviour tables live.
SECONDARY_FILTERS = [
    {"scope": "raw", "datatype": "beh", "suffix": "beh", "extension": ".tsv"},
    {"scope": "raw", "datatype": "ieeg", "suffix": "electrodes", "extension": ".tsv"},
]

PARAMS = TrialStatsParams(
    anchor_event_codes=["10"],
    tmin_s=-2.0,
    tmax_s=10.0,
    condition_a="accepted",
    condition_b="rejected",
    #atlas_name="MarsAtlas",
    #n_bins=24,
    p_value_correction_method="fdr_bh",
    significance_alpha=0.05,
)

# This resolver is the task-specific layer for accepted vs rejected.
# Update the column names and label map to match your dataset.
RESOLVER = TableTrialLabelResolver(
    label_column="choice",
    label_map={
        "0": "rejected",
        "1": "accepted",
    },
)

WRITER_PARAMS = TrialStatsWriterParams(
    bids_root=BIDS_ROOT,
    output_format="hdf5"
)

N_JOBS = 1


def _build_subject_groups(dataset: BIDSDataset) -> list[BIDSFileGroup]:
    ieeg_files = sorted(
        dataset.get_files(**IEEG_FILTERS),
        key=lambda file: str(file.path),
    )
    secondary_files = _secondary_files(dataset)

    groups: list[BIDSFileGroup] = []
    for subject_id in sorted({file.get("subject") for file in ieeg_files}):
        subject_ieeg = [file for file in ieeg_files if file.get("subject") == subject_id]
        if not subject_ieeg:
            continue
        subject_secondaries = [
            file
            for file in secondary_files
            if file.get("subject") == subject_id
        ]
        groups.append(
            BIDSFileGroup(
                primary=subject_ieeg[0],
                secondaries=subject_ieeg[1:] + subject_secondaries,
            )
        )

    return groups


def _secondary_files(dataset: BIDSDataset) -> list[BIDSFile]:
    files_by_path: dict[Path, BIDSFile] = {}
    for entity_filters in SECONDARY_FILTERS:
        for file in dataset.get_files(**entity_filters):
            files_by_path[file.path] = file
    return sorted(files_by_path.values(), key=lambda file: str(file.path))


if __name__ == "__main__":
    ds = BIDSDataset(BIDS_ROOT)
    groups = _build_subject_groups(ds)
    print(f"Found {len(groups)} subject group(s). Running with n_jobs={N_JOBS}.")

    processor = TrialStatsProcessing(PARAMS, resolver=RESOLVER)
    writer = TrialStatsProcessingWriter(WRITER_PARAMS)

    out_paths = processor.run(groups, writer, n_jobs=N_JOBS)
    for path in out_paths:
        print(f"Wrote {path}")

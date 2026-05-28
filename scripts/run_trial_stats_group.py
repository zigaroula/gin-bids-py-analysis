"""
Group-level ROI statistics on condition_test outputs - run script.
Edit the parameters below and run: python scripts/run_trial_stats_group.py
"""

from __future__ import annotations

from pathlib import Path

from bidsforge.bids import BIDSDataset, BIDSFile, BIDSFileGroup
from bidsforge.processing.trial_stats_group import (
    ConditionTestGroupParams,
    ConditionTestGroupProcessing,
    ConditionTestGroupProcessingWriter,
    ConditionTestGroupWriterParams,
    build_condition_test_compatible_groups,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

BIDS_ROOT = Path(r"E:\CBT\bids")

# Query condition_test channel-level outputs from derivatives/condition_test.
TRIAL_STATS_FILTERS = {
    "scope": "condition_test",
    "suffix": "stats",
    "extension": ".h5",
    # "task": "decid",
}

PARAMS = ConditionTestGroupParams(
    primary_condition_metric="t_values",
    p_value_correction_method="none",
    cluster_permutation_method="mne",
    significance_alpha=0.05,
    roi_mode="manual",
    manual_region_channels={
        "daINS": {
            "epi01": ["Y02", "Y06"],
            "epi03": ["IAD2"],
            "epi05": ["X04", "X07", "X03", "X06"],
            "epi07": ["X08", "T03"],
            "epi11": ["X05", "X07", "X06"],
            "epi12": ["Ap02"],
            "epi14": ["Xp04"],
            "epi17": ["II8"],
            "epi18": ["X06"],
            "epi19": ["IMD2"],
            "epi21": ["XS7", "XD7", "XD4", "XD8"],
            "epi22": ["XS8"],
            "epi23": ["EL2"]
        },
        "vaINS": {
            "epi04": ["Bp02"],
            "epi08": ["IA2", "IA4"],
            "epi11": ["X04"],
            "epi14": ["Xp02"],
            "epi17": ["II2"],
            "epi18": ["Y02", "X02"],
            "epi21": ["XS4"],
            "epi22": ["XS2", "YS2"]
        },
    },
)

WRITER_PARAMS = ConditionTestGroupWriterParams(
    bids_root=BIDS_ROOT,
    output_format="hdf5",
    output_description="none"
)

N_JOBS = 1


def _load_trial_stats_files(dataset: BIDSDataset) -> list[BIDSFile]:
    files = dataset.get_files(**TRIAL_STATS_FILTERS)
    return sorted(files, key=lambda file: str(file.path))


def _build_groups(files: list[BIDSFile], params: ConditionTestGroupParams) -> list[BIDSFileGroup]:
    return build_condition_test_compatible_groups(
        files,
        primary_condition_metric=params.primary_condition_metric,
    )


def main() -> list[Path]:
    ds = BIDSDataset(BIDS_ROOT)
    files = _load_trial_stats_files(ds)
    groups = _build_groups(files, PARAMS)
    print(
        f"Found {len(files)} condition_test file(s) grouped into {len(groups)} compatible run(s). "
        f"Running with n_jobs={N_JOBS}."
    )

    if not groups:
        return []

    processor = ConditionTestGroupProcessing(PARAMS)
    writer = ConditionTestGroupProcessingWriter(WRITER_PARAMS)
    out_paths = processor.run(groups, writer, n_jobs=N_JOBS)
    for path in out_paths:
        print(f"Wrote {path}")
    return out_paths


if __name__ == "__main__":
    main()


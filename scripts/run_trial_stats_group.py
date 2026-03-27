"""
Group-level ROI statistics on trial_stats outputs - run script.
Edit the parameters below and run: python scripts/run_trial_stats_group.py
"""

from __future__ import annotations

from pathlib import Path

from gin_bids_py_analysis.bids import BIDSDataset, BIDSFile, BIDSFileGroup
from gin_bids_py_analysis.processing.trial_stats_group import (
    TrialStatsGroupParams,
    TrialStatsGroupProcessing,
    TrialStatsGroupProcessingWriter,
    TrialStatsGroupWriterParams,
    build_trial_stats_compatible_groups,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

BIDS_ROOT = Path(r"D:\CBT\bids")

# Query trial_stats channel-level outputs from derivatives/trial_stats.
TRIAL_STATS_FILTERS = {
    "scope": "trial_stats",
    "suffix": "stats",
    "extension": ".h5",
    # "task": "decid",
}

PARAMS = TrialStatsGroupParams(
    source_metric="t_values",
    p_value_correction_method="none",
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
            "epi17": ["II08"],
            "epi18": ["X06"],
            "epi19": ["IMD2"],
            "epi21": ["XS07", "XD07", "XD04", "XD08"],
            "epi22": ["XS08"],
            "epi23": ["EL02"]
        },
        "vaINS": {
            "epi04": ["Bp2"],
            "epi08": ["IA2", "IA4"],
            "epi11": ["X04"],
            "epi14": ["Xp02"],
            "epi17": ["II02"],
            "epi18": ["Y02", "X02"],
            "epi21": ["XS04"],
            "epi22": ["XS02", "YS02"]
        },
    }
)

WRITER_PARAMS = TrialStatsGroupWriterParams(
    bids_root=BIDS_ROOT,
    output_format="hdf5"
)

N_JOBS = 1


def _load_trial_stats_files(dataset: BIDSDataset) -> list[BIDSFile]:
    files = dataset.get_files(**TRIAL_STATS_FILTERS)
    return sorted(files, key=lambda file: str(file.path))


def _build_groups(files: list[BIDSFile], params: TrialStatsGroupParams) -> list[BIDSFileGroup]:
    return build_trial_stats_compatible_groups(
        files,
        source_metric=params.source_metric,
    )


def main() -> list[Path]:
    ds = BIDSDataset(BIDS_ROOT)
    files = _load_trial_stats_files(ds)
    groups = _build_groups(files, PARAMS)
    print(
        f"Found {len(files)} trial_stats file(s) grouped into {len(groups)} compatible run(s). "
        f"Running with n_jobs={N_JOBS}."
    )

    if not groups:
        return []

    processor = TrialStatsGroupProcessing(PARAMS)
    writer = TrialStatsGroupProcessingWriter(WRITER_PARAMS)
    out_paths = processor.run(groups, writer, n_jobs=N_JOBS)
    for path in out_paths:
        print(f"Wrote {path}")
    return out_paths


if __name__ == "__main__":
    main()

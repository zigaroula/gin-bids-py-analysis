"""
Group-level ROI statistics on regression outputs - run script.
Edit the parameters below and run: python scripts/run_trial_slope_stats_group.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from gin_bids_py_analysis.bids import BIDSDataset, BIDSFile, BIDSFileGroup
from gin_bids_py_analysis.bids.helpers import normalize_subject_value
from gin_bids_py_analysis.processing.trial_slope_stats_group import (
    TrialSlopeStatsGroupParams,
    TrialSlopeStatsGroupProcessing,
    TrialSlopeStatsGroupProcessingWriter,
    TrialSlopeStatsGroupWriterParams,
    build_trial_slope_stats_compatible_groups,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

BIDS_ROOT = Path(r"E:\data_clarissa\valuation\bids")

# Kept aligned with scripts/run_trial_slope_stats.py inputs.
IEEG_FILTERS = {
    "suffix": "ieeg",
    "extension": ".vhdr",
    "desc": "gammasm250",
}

# Kept aligned with scripts/run_trial_slope_stats.py inputs.
SECONDARY_FILTERS = [
    {"scope": "raw", "datatype": "beh", "suffix": "beh", "extension": ".tsv"},
    {"scope": "raw", "datatype": "ieeg", "suffix": "electrodes", "extension": ".tsv"},
]

# Query regression channel-level outputs from derivatives/regression.
# desc must match RegressionWriterParams(output_description=...) used upstream.
TRIAL_SLOPE_STATS_FILTERS = {
    "scope": "regression",
    "suffix": "stats",
    "extension": ".h5",
    "desc": "correlation",
    # "task": "decid",
}

ROI_CSV_FILES = {
    "vmPFC": Path(r"C:\Users\Benjamin BONTEMPS\Downloads\PFCvm_elecs_tbl.csv"),
    "daINS": Path(r"C:\Users\Benjamin BONTEMPS\Downloads\aINS_dors_elecs_tbl.csv"),
    "vaINS": Path(r"C:\Users\Benjamin BONTEMPS\Downloads\aINS_vent_elecs_tbl.csv"),
}

GROUP_PARAM_KWARGS = {
    "p_value_correction_method": "none",
    "significance_alpha": 0.05,
    "roi_mode": "manual",
}

WRITER_PARAMS = TrialSlopeStatsGroupWriterParams(
    bids_root=BIDS_ROOT,
    output_format="hdf5",
    output_description="none",
)

N_JOBS = 1
_NA_LIKE_TOKENS = frozenset({"nan", "na", "n/a", "none", "null"})
_FIRST_CONTACT_PATTERN = re.compile(r"^([A-Za-z]+[0-9]+)")


def _is_nan_like(value: object) -> bool:
    return str(value).strip().casefold() in _NA_LIKE_TOKENS


def _normalize_subject_from_csv(raw_subject: object) -> str:
    subject = str(raw_subject).strip().replace("_", "")
    return normalize_subject_value(subject)


def _extract_first_bipolar_contact(raw_channel: object) -> str:
    channel = str(raw_channel).strip()
    if not channel:
        return ""
    match = _FIRST_CONTACT_PATTERN.match(channel)
    if match is not None:
        return match.group(1)
    for separator in ("-", "_", " "):
        if separator in channel:
            return channel.split(separator, 1)[0].strip()
    return channel


def _row_contains_nan(row: dict[str, object]) -> bool:
    return any(_is_nan_like(value) for value in row.values())


def _load_roi_channels_from_csv(csv_paths_by_roi: dict[str, Path]) -> dict[str, dict[str, list[str]]]:
    manual_region_channels: dict[str, dict[str, list[str]]] = {}

    for roi_name, csv_path in csv_paths_by_roi.items():
        if not csv_path.exists():
            raise FileNotFoundError(
                f"ROI CSV not found for {roi_name!r}: {csv_path}"
            )

        with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            if not reader.fieldnames or len(reader.fieldnames) < 2:
                raise ValueError(
                    f"{csv_path}: expected at least 2 columns (subject, channel)."
                )
            subject_col = reader.fieldnames[0]
            channel_col = reader.fieldnames[1]

            roi_subject_channels: dict[str, list[str]] = {}
            seen_channels: dict[str, set[str]] = {}
            for row in reader:
                if _row_contains_nan(row):
                    continue

                subject = _normalize_subject_from_csv(row.get(subject_col, ""))
                channel = _extract_first_bipolar_contact(row.get(channel_col, ""))
                if not subject or not channel:
                    continue

                subject_seen = seen_channels.setdefault(subject, set())
                normalized_channel = channel.casefold()
                if normalized_channel in subject_seen:
                    continue
                subject_seen.add(normalized_channel)
                roi_subject_channels.setdefault(subject, []).append(channel)

            if roi_subject_channels:
                manual_region_channels[roi_name] = roi_subject_channels

    if not manual_region_channels:
        raise ValueError(
            "No ROI channels were loaded from CSV files after filtering NaN rows."
        )

    return manual_region_channels


def _build_group_params() -> TrialSlopeStatsGroupParams:
    manual_region_channels = _load_roi_channels_from_csv(ROI_CSV_FILES)
    return TrialSlopeStatsGroupParams(
        manual_region_channels=manual_region_channels,
        **GROUP_PARAM_KWARGS,
    )


def _print_roi_summary(manual_region_channels: dict[str, dict[str, list[str]]]) -> None:
    for roi_name, subject_map in manual_region_channels.items():
        n_subjects = len(subject_map)
        n_channels = sum(len(channels) for channels in subject_map.values())
        print(f"ROI {roi_name}: {n_channels} channel(s) across {n_subjects} subject(s).")


def _load_trial_slope_stats_files(dataset: BIDSDataset) -> list[BIDSFile]:
    files = dataset.get_files(**TRIAL_SLOPE_STATS_FILTERS)
    return sorted(files, key=lambda file: str(file.path))


def _build_groups(files: list[BIDSFile]) -> list[BIDSFileGroup]:
    return build_trial_slope_stats_compatible_groups(files)


def main() -> list[Path]:
    params = _build_group_params()
    _print_roi_summary(params.manual_region_channels)

    ds = BIDSDataset(BIDS_ROOT)
    files = _load_trial_slope_stats_files(ds)
    groups = _build_groups(files)
    print(
        f"Found {len(files)} regression file(s) grouped into {len(groups)} "
        f"compatible run(s). Running with n_jobs={N_JOBS}."
    )

    if not groups:
        return []

    processor = TrialSlopeStatsGroupProcessing(params)
    writer = TrialSlopeStatsGroupProcessingWriter(WRITER_PARAMS)
    out_paths = processor.run(groups, writer, n_jobs=N_JOBS)
    for path in out_paths:
        print(f"Wrote {path}")
    return out_paths


if __name__ == "__main__":
    main()

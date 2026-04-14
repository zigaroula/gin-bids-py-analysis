"""
Trial slope statistics visualization script.
Edit parameters below, then run:

    python scripts/visualize_trial_slope_stats.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from gin_bids_py_analysis.bids import BIDSDataset, BIDSFileGroup, build_subject_groups
from gin_bids_py_analysis.bids.helpers import normalize_subject_value
from gin_bids_py_analysis.processing.trial_stats.regression import RegressionParams
from gin_bids_py_analysis.processing.trial_stats_group import RegressionGroupParams
from gin_bids_py_analysis.processing.utils.trial_resolver import TableTrialResolver
from gin_bids_py_analysis.visualization.trial_stats import launch_slope

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

# Kept aligned with scripts/run_trial_slope_stats.py inputs.
BIDS_ROOT = Path(r"D:\data_clarissa\valuation\bids")

IEEG_FILTERS = {
    "suffix": "ieeg",
    "extension": ".vhdr",
    "desc": "gammasm250",
}

SECONDARY_FILTERS = [
    {"scope": "raw", "datatype": "beh", "suffix": "beh", "extension": ".tsv"},
    {"scope": "raw", "datatype": "ieeg", "suffix": "electrodes", "extension": ".tsv"},
]

PARAMS = RegressionParams(
    anchor_event_codes=["11", "12"],
    experiment_start_event_code="5",
    tmin_s=-0.5,
    tmax_s=5.0,
    condition_a="pleasant",
    condition_b="unpleasant",
    predictor="rating",
    predictor_transform_by_condition={
        "pleasant": {"scale": 1.0, "offset": 0.0},
        "unpleasant": {"scale": -1.0, "offset": 0.0},
    },
    predictor_zscore="none",
    activity_zscore="baseline",
    activity_baseline_tmin_s=-0.25,
    activity_baseline_tmax_s=-0.05,
    activity_baseline_scope="global",
    activity_baseline_remove_outlier_trial_means=True,
    p_value_correction_method="none",
    significance_alpha=0.05,
    trial_activity_summary={
        "kind": "anchor_to_response_mean",
        "response": {"source": "table_column", "column": "RT", "units": "s"},
    },
    epoch_cleaning={
        "reject_trials_by_epoch_mean": True,
        "reject_trials_by_epoch_max": True,
        "reject_channels_by_trial_mean_spread": True,
        "reject_channels_by_trial_max_spread": True,
        "max_nan_trial_ratio": 0.25,
    }
)

RESOLVER = TableTrialResolver(
    conditions=[
        {
            "label": "pleasant",
            "when": {
                "all": [
                    {"column": "pleasant", "op": "==", "value": 1},
                    {"column": "rating", "op": ">=", "value": 0},
                ]
            },
        },
        {
            "label": "unpleasant",
            "when": {
                "all": [
                    {"column": "pleasant", "op": "==", "value": 2},
                    {"column": "rating", "op": ">=", "value": 0},
                ]
            },
        },
    ],
    extract_columns=["rating", "RT"],
)

ROI_CSV_FILES = {
    "vmPFC": Path(r"D:\data_clarissa\valuation\csv\PFCvm_elecs_tbl.csv"),
    "daINS": Path(r"D:\data_clarissa\valuation\csv\aINS_dors_elecs_tbl.csv"),
    "vaINS": Path(r"D:\data_clarissa\valuation\csv\aINS_vent_elecs_tbl.csv"),
}

GROUP_PARAM_KWARGS = {
    "p_value_correction_method": "none",
    "significance_alpha": 0.05,
    "roi_mode": "manual",
}

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
                channel_key = channel.casefold()
                if channel_key in subject_seen:
                    continue
                subject_seen.add(channel_key)
                roi_subject_channels.setdefault(subject, []).append(channel)

            if roi_subject_channels:
                manual_region_channels[roi_name] = roi_subject_channels

    if not manual_region_channels:
        raise ValueError(
            "No ROI channels were loaded from CSV files after filtering NaN rows."
        )

    return manual_region_channels


def _build_group_params() -> RegressionGroupParams:
    manual_region_channels = _load_roi_channels_from_csv(ROI_CSV_FILES)
    return RegressionGroupParams(
        manual_region_channels=manual_region_channels,
        **GROUP_PARAM_KWARGS,
    )


def _print_roi_summary(manual_region_channels: dict[str, dict[str, list[str]]]) -> None:
    for roi_name, subject_map in manual_region_channels.items():
        n_subjects = len(subject_map)
        n_channels = sum(len(channels) for channels in subject_map.values())
        print(f"ROI {roi_name}: {n_channels} channel(s) across {n_subjects} subject(s).")


def _build_subject_groups(dataset: BIDSDataset) -> dict[str, BIDSFileGroup]:
    groups = build_subject_groups(dataset, IEEG_FILTERS, SECONDARY_FILTERS)
    return {group.primary.get("subject"): group for group in groups}


if __name__ == "__main__":
    group_params = _build_group_params()
    _print_roi_summary(group_params.manual_region_channels)

    ds = BIDSDataset(BIDS_ROOT)
    subject_groups = _build_subject_groups(ds)
    print(f"Found {len(subject_groups)} subject(s).")

    launch_slope(
        subject_groups,
        PARAMS,
        RESOLVER,
        group_params=group_params,
        bids_root=BIDS_ROOT,
    )


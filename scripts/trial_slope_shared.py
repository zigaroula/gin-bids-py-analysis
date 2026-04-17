from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

from gin_bids_py_analysis.bids import BIDSDataset, BIDSFileGroup, build_subject_groups
from gin_bids_py_analysis.bids.helpers import normalize_subject_value
from gin_bids_py_analysis.processing.trial_stats.regression import RegressionParams
from gin_bids_py_analysis.processing.trial_stats.result import BaseTrialStatsProcessingResult
from gin_bids_py_analysis.processing.trial_stats_group import RegressionGroupParams
from gin_bids_py_analysis.processing.utils.condition_rules import ConditionExpr
from gin_bids_py_analysis.processing.utils.trial_annotator import (
    EventAnnotationInvalidationRule,
    EventFileWindowAnnotator,
)
from gin_bids_py_analysis.processing.utils.trial_resolver import TableTrialResolver

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

BIDS_ROOT = Path(r"D:\data_clarissa\valuation\bids")

IEEG_FILTERS = {
    "suffix": "ieeg",
    "extension": ".vhdr",
    "desc": "gammasm250",
}

SECONDARY_FILTERS = [
    {"scope": "raw", "datatype": "beh", "suffix": "beh", "extension": ".tsv"},
    {"scope": "raw", "datatype": "ieeg", "suffix": "electrodes", "extension": ".tsv"},
    {
        "scope": "delphos",
        "datatype": "ieeg",
        "suffix": "events",
        "extension": ".tsv",
        "desc": "delphos",
    },
]

TRIAL_SLOPE_EPOCH_CLEANING = {
    "reject_trials_by_epoch_mean": True,
    "reject_trials_by_epoch_max": True,
    "reject_by_trial_mean_spread": True,
    "reject_by_trial_max_spread": True,
    "max_nan_trial_ratio": 0.25,
}

TRIAL_SLOPE_EXTRACT_COLUMNS = ["rating", "RT"]

TRIAL_SLOPE_RESOLVER_CONDITIONS = [
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
    epoch_cleaning=TRIAL_SLOPE_EPOCH_CLEANING,
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

VM_PFC_SPIKE_EXCLUSION_REASON = "vmPFC_spike_0_5s"

_NA_LIKE_TOKENS = frozenset({"nan", "na", "n/a", "none", "null"})
_FIRST_CONTACT_PATTERN = re.compile(r"^([A-Za-z]+[0-9]+)")


def build_trial_slope_resolver() -> TableTrialResolver:
    return TableTrialResolver(
        conditions=TRIAL_SLOPE_RESOLVER_CONDITIONS,
        extract_columns=TRIAL_SLOPE_EXTRACT_COLUMNS,
        filter={"suffix": "beh"},
    )


RESOLVER = build_trial_slope_resolver()


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


def load_roi_channels_from_csv(
    csv_paths_by_roi: dict[str, Path] = ROI_CSV_FILES,
) -> dict[str, dict[str, list[str]]]:
    manual_region_channels: dict[str, dict[str, list[str]]] = {}

    for roi_name, csv_path in csv_paths_by_roi.items():
        if not csv_path.exists():
            raise FileNotFoundError(f"ROI CSV not found for {roi_name!r}: {csv_path}")

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


def print_roi_summary(manual_region_channels: dict[str, dict[str, list[str]]]) -> None:
    for roi_name, subject_map in manual_region_channels.items():
        n_subjects = len(subject_map)
        n_channels = sum(len(channels) for channels in subject_map.values())
        print(f"ROI {roi_name}: {n_channels} channel(s) across {n_subjects} subject(s).")


def build_vmPFC_spike_filter(
    vm_pfc_channels_by_subject: dict[str, list[str]],
) -> ConditionExpr:
    per_subject_filters: list[ConditionExpr] = []

    for subject_id, channels in sorted(vm_pfc_channels_by_subject.items()):
        unique_channels: list[str] = []
        seen_channels: set[str] = set()
        for channel in channels:
            cleaned = str(channel).strip()
            if not cleaned:
                continue
            channel_key = cleaned.casefold()
            if channel_key in seen_channels:
                continue
            seen_channels.add(channel_key)
            unique_channels.append(cleaned)

        if not unique_channels:
            continue

        per_subject_filters.append(
            ConditionExpr(
                all=[
                    ConditionExpr(column="subject", op="==", value=subject_id),
                    ConditionExpr(column="event_type", op="==", value="Spike"),
                    ConditionExpr(column="channel", op="in", values=unique_channels),
                ]
            )
        )

    if not per_subject_filters:
        raise ValueError("No vmPFC channels available to build the Delphos spike filter.")

    if len(per_subject_filters) == 1:
        return per_subject_filters[0]

    return ConditionExpr(any=per_subject_filters)


def build_trial_annotators(
    manual_region_channels: dict[str, dict[str, list[str]]],
) -> list[EventFileWindowAnnotator | EventAnnotationInvalidationRule]:
    vm_pfc_channels_by_subject = manual_region_channels.get("vmPFC", {})
    if not vm_pfc_channels_by_subject:
        raise ValueError("vmPFC ROI channels are required to exclude Delphos spike trials.")

    return [
        EventFileWindowAnnotator(
            filter={"suffix": "events", "desc": "delphos"},
            metadata_events_key="delphos_events",
            window_tmin_s=0.0,
            window_tmax_s=5.0,
        ),
        EventAnnotationInvalidationRule(
            metadata_events_key="delphos_events",
            event_filter=build_vmPFC_spike_filter(vm_pfc_channels_by_subject),
            exclusion_reason=VM_PFC_SPIKE_EXCLUSION_REASON,
        ),
    ]


def build_group_params(
    manual_region_channels: dict[str, dict[str, list[str]]] | None = None,
) -> RegressionGroupParams:
    channels_by_roi = (
        manual_region_channels
        if manual_region_channels is not None
        else load_roi_channels_from_csv()
    )
    return RegressionGroupParams(
        manual_region_channels=channels_by_roi,
        **GROUP_PARAM_KWARGS,
    )


def build_trial_slope_groups(
    dataset: BIDSDataset,
) -> list[BIDSFileGroup]:
    return build_subject_groups(dataset, IEEG_FILTERS, SECONDARY_FILTERS)


def build_trial_slope_subject_groups(
    dataset: BIDSDataset,
) -> dict[str, BIDSFileGroup]:
    return {
        group.primary.get("subject"): group
        for group in build_trial_slope_groups(dataset)
    }


def count_excluded_trials_by_reason(
    trials: list[Any],
    exclusion_reason: str = VM_PFC_SPIKE_EXCLUSION_REASON,
) -> int:
    return sum(
        1
        for trial in trials
        if not getattr(trial, "keep", True)
        and getattr(trial, "exclusion_reason", None) == exclusion_reason
    )


def format_subject_trial_exclusion_summary(
    subject_id: str,
    result: BaseTrialStatsProcessingResult,
    *,
    exclusion_reason: str = VM_PFC_SPIKE_EXCLUSION_REASON,
) -> str:
    removed_for_reason = count_excluded_trials_by_reason(
        result.resolved_trials,
        exclusion_reason,
    )
    excluded_total = sum(1 for trial in result.resolved_trials if not trial.keep)
    total_resolved = len(result.resolved_trials)
    nan_masked_count = sum(
        1 for trial in result.resolved_trials if trial.metadata.get("nan_masked_features")
    )
    dropped_channels = len(result.epoch_cleaning_audit.get("excluded_features", {}))
    return (
        f"Subject {subject_id}: removed {removed_for_reason} trial(s) by "
        f"{exclusion_reason} ({excluded_total} excluded total / "
        f"{total_resolved} resolved) | "
        f"NaN-masked: {nan_masked_count} trial(s) | "
        f"channels dropped: {dropped_channels}."
    )


def print_subject_trial_exclusion_summary(
    subject_id: str,
    result: BaseTrialStatsProcessingResult,
) -> None:
    print(format_subject_trial_exclusion_summary(subject_id, result))

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Any

import numpy as np

from gin_bids_py_analysis.bids import BIDSDataset, BIDSFile, BIDSFileGroup, build_subject_groups
from gin_bids_py_analysis.bids.helpers import normalize_subject_value
from gin_bids_py_analysis.processing.trial_stats.regression import RegressionParams
from gin_bids_py_analysis.processing.trial_stats.result import BaseTrialStatsProcessingResult
from gin_bids_py_analysis.processing.trial_stats_group import RegressionGroupParams
from gin_bids_py_analysis.processing.utils.condition_rules import ConditionExpr
from gin_bids_py_analysis.processing.utils.trial_annotator import (
    EventAnnotationInvalidationRule,
    EventFileWindowAnnotator,
    TrialMetadataInvalidationRule,
)
from gin_bids_py_analysis.processing.utils.trial_resolver import ResolvedTrial, TableTrialResolver

# ---------------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------------

BIDS_ROOT = Path(r"E:/data_clarissa/valuation/bids")

# ---------------------------------------------------------------------------
# MATLAB z-score injection configuration
# ---------------------------------------------------------------------------
# Set USE_MATLAB_ZSCORES to True to replace the default 'rating' predictor
# with pre-computed z-scored values loaded from an external MATLAB file.
# When enabled:
#  - Z-scores are loaded from MATLAB_ZSCORES_PATH and injected as metadata["matlab_zscore"]
#  - The predictor is automatically changed to "matlab_zscore"
#  - Behavioral filters (RT, rating thresholds) still apply to RAW rating values
#  - Trials are matched by sequential order of appearance per subject
# Set to False to restore default behavior (predictor="rating").

USE_MATLAB_ZSCORES: bool = False

MATLAB_ZSCORES_PATH = Path(r"E:/data_clarissa/subjects.mat")
MATLAB_ZSCORE_COLUMN_INDEX = 6  # 0-based index for column 7 in trial_characteristics1

IEEG_FILTERS = {
    "suffix": "ieeg",
    "extension": ".vhdr",
    "desc": "bgasm250",
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
            ]
        },
    },
    {
        "label": "unpleasant",
        "when": {
            "all": [
                {"column": "pleasant", "op": "==", "value": 2},
            ]
        },
    },
]


def _build_params() -> RegressionParams:
    """Build RegressionParams with conditional predictor based on USE_MATLAB_ZSCORES."""
    return RegressionParams(
        anchor_event_codes=["11", "12"],
        experiment_start_event_code="5",
        tmin_s=-1,
        tmax_s=10,
        condition_a="pleasant",
        condition_b="unpleasant",
        predictor="matlab_zscore" if USE_MATLAB_ZSCORES else "rating",
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
        activity_baseline_outlier_method="median_mad",
        p_value_correction_method="none",
        significance_alpha=0.05,
        trial_activity_summary={
            "kind": "anchor_to_response_mean",
            "response": {"source": "table_column", "column": "RT", "units": "s"},
        },
        epoch_cleaning=TRIAL_SLOPE_EPOCH_CLEANING,
        #n_permutations=500
    )


PARAMS = _build_params()

ROI_CSV_FILES = {
    "vmPFC": Path(r"E:/data_clarissa/valuation/csv/PFCvm_elecs_tbl.csv"),
    "daINS": Path(r"E:/data_clarissa/valuation/csv/aINS_dors_elecs_tbl.csv"),
    "vaINS": Path(r"E:/data_clarissa/valuation/csv/aINS_vent_elecs_tbl.csv"),
}

GROUP_PARAM_KWARGS = {
    "p_value_correction_method": "none",
    "significance_alpha": 0.05,
    "roi_mode": "manual",
}

VM_PFC_SPIKE_EXCLUSION_REASON = "vmPFC_spike_0_5s"

# Behavioral thresholds applied as annotator invalidation rules (orthogonal to
# condition classification).  These replicate MATLAB b2:
#   opts.removeoutlierRTs  → trials with RT > MAX_RT_S are excluded (all channels)
#   opts.removenegratings  → trials with rating < 0 are excluded (all channels)
MAX_RT_S: float = 20.0
MIN_RATING: float = 0.0
BEH_TSV_PATH = Path(
    r"E:/data_clarissa/valuation/bids"
    r"/sub-GRE2021AICb/beh/sub-GRE2021AICb_task-MDCHOICE_beh.tsv"
)

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


def load_matlab_zscores(mat_path: Path) -> dict[str, np.ndarray]:
    """Load z-scored predictor values from MATLAB subjects.mat file.

    Parameters
    ----------
    mat_path : Path
        Path to the MATLAB .mat file containing subject trial characteristics.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping normalized BIDS subject IDs to z-score arrays.
        Subject names are normalized (underscores removed) to match BIDS format.

    Raises
    ------
    FileNotFoundError
        If mat_path does not exist.
    ValueError
        If the .mat file structure is invalid or missing expected fields.

    Notes
    -----
    The function expects the .mat file to contain a 'subjects' array with fields:
    - 'name': subject identifier (underscores will be removed for BIDS matching)
    - 'trial_characteristics1': (n_trials, 7) array where column 7 (index 6)
      contains the z-scored predictor values.
    """
    if not mat_path.exists():
        raise FileNotFoundError(f"MATLAB z-scores file not found: {mat_path}")

    try:
        from scipy.io import loadmat  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "scipy is required to load MATLAB files. Install with: pip install scipy"
        ) from exc

    try:
        mat = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    except Exception as exc:
        raise ValueError(f"Failed to load MATLAB file {mat_path}: {exc}") from exc

    if "subjects" not in mat:
        raise ValueError(f"MATLAB file {mat_path} does not contain 'subjects' field.")

    subjects_data = mat["subjects"]
    if not hasattr(subjects_data, "__iter__"):
        subjects_data = [subjects_data]

    zscores_by_subject: dict[str, np.ndarray] = {}

    for subject_obj in subjects_data:
        if not hasattr(subject_obj, "name"):
            continue

        raw_name = str(subject_obj.name).strip()
        if not raw_name:
            continue

        # Normalize subject name: remove underscores to match BIDS format
        normalized_name = normalize_subject_value(raw_name.replace("_", ""))

        if not hasattr(subject_obj, "trial_characteristics1"):
            raise ValueError(
                f"Subject {raw_name} is missing 'trial_characteristics1' field."
            )

        tc = subject_obj.trial_characteristics1
        if not isinstance(tc, np.ndarray):
            tc = np.asarray(tc, dtype=np.float64)

        if tc.ndim != 2:
            raise ValueError(
                f"Subject {raw_name}: trial_characteristics1 must be 2D, got shape {tc.shape}."
            )

        if tc.shape[1] <= MATLAB_ZSCORE_COLUMN_INDEX:
            raise ValueError(
                f"Subject {raw_name}: trial_characteristics1 has only {tc.shape[1]} columns, "
                f"cannot extract column {MATLAB_ZSCORE_COLUMN_INDEX + 1}."
            )

        zscores = tc[:, MATLAB_ZSCORE_COLUMN_INDEX].astype(np.float64)
        zscores_by_subject[normalized_name] = zscores

    if not zscores_by_subject:
        raise ValueError(f"No valid subjects found in MATLAB file {mat_path}.")

    return zscores_by_subject


class MatlabZscorePredictorAnnotator:
    """Inject MATLAB-derived z-scored predictor values into trial metadata.

    This annotator loads z-scores from an external MATLAB file and injects them
    into ``trial.metadata["matlab_zscore"]`` for each resolved trial. Trials are
    matched by sequential order of appearance (index) per subject.

    Trials that are already excluded (``keep=False``) or fall outside the available
    z-score range receive ``np.nan``.

    This allows substituting the default predictor (e.g., 'rating') with pre-computed
    z-scored values while preserving behavioral filtering on the original raw values.

    Parameters
    ----------
    mat_path : Path
        Path to the MATLAB .mat file. Loaded once at construction.

    Attributes
    ----------
    zscores_by_subject : dict[str, np.ndarray]
        Pre-loaded z-score arrays indexed by normalized BIDS subject ID.

    Notes
    -----
    - Implements the ``TrialWindowAnnotator`` protocol.
    - Z-scores are matched by trial index within each subject's kept trials.
    - Missing subjects or out-of-range trial indices result in NaN values.
    - Does not modify ``trial.keep`` or ``trial.exclusion_reason``.
    """

    def __init__(self, mat_path: Path) -> None:
        self.zscores_by_subject = load_matlab_zscores(mat_path)

    def annotate_trials(
        self,
        group: BIDSFileGroup,
        ieeg_file: BIDSFile,
        trials: list[ResolvedTrial],
        tmin_s: float,
        tmax_s: float,
        *,
        ieeg_channel_names: list[str],
    ) -> None:
        """Inject matlab_zscore into trial metadata by sequential order.

        Parameters
        ----------
        group : BIDSFileGroup
            File group being processed (unused).
        ieeg_file : BIDSFile
            Current iEEG file. Subject ID is extracted from this file.
        trials : list[ResolvedTrial]
            Resolved trials to annotate. Modified in-place.
        tmin_s : float
            Epoch start time (unused).
        tmax_s : float
            Epoch end time (unused).
        ieeg_channel_names : list[str]
            Available channel names (unused).
        """
        del group, tmin_s, tmax_s, ieeg_channel_names  # Unused

        subject_id = ieeg_file.get("subject", "")
        if not subject_id:
            # No subject ID available; mark all trials with NaN
            for trial in trials:
                trial.metadata["matlab_zscore"] = np.nan
            return

        zscores = self.zscores_by_subject.get(subject_id)
        if zscores is None:
            # Subject not found in MATLAB file; mark all trials with NaN
            for trial in trials:
                trial.metadata["matlab_zscore"] = np.nan
            return

        # Match z-scores by sequential index of kept trials
        kept_idx = 0
        for trial in trials:
            if not trial.keep:
                trial.metadata["matlab_zscore"] = np.nan
                continue

            if kept_idx < len(zscores):
                trial.metadata["matlab_zscore"] = float(zscores[kept_idx])
            else:
                # Out of range; use NaN
                trial.metadata["matlab_zscore"] = np.nan

            kept_idx += 1


def build_trial_annotators(
    manual_region_channels: dict[str, dict[str, list[str]]],
) -> list:
    """Build the list of trial annotators for trial slope processing.

    When USE_MATLAB_ZSCORES is True, a MatlabZscorePredictorAnnotator is prepended
    to inject z-scored predictor values before behavioral filtering.

    Parameters
    ----------
    manual_region_channels : dict[str, dict[str, list[str]]]
        ROI channel mapping by subject, used for vmPFC spike exclusion.

    Returns
    -------
    list
        List of annotators including optional MATLAB z-score injection,
        behavioral thresholds, and Delphos spike filtering.

    Raises
    ------
    ValueError
        If vmPFC ROI channels are not provided.
    """
    vm_pfc_channels_by_subject = manual_region_channels.get("vmPFC", {})
    if not vm_pfc_channels_by_subject:
        raise ValueError("vmPFC ROI channels are required to exclude Delphos spike trials.")

    annotators: list = []

    # Optionally inject MATLAB z-scores BEFORE behavioral filtering
    if USE_MATLAB_ZSCORES:
        annotators.append(MatlabZscorePredictorAnnotator(MATLAB_ZSCORES_PATH))

    annotators.extend([
        # Behavioral validity: exclude trials with RT > MAX_RT_S or rating < MIN_RATING.
        # These checks are orthogonal to condition classification and mirror MATLAB b2
        # steps opts.removeoutlierRTs and opts.removenegratings.
        # NOTE: These filters still apply to RAW rating values even when USE_MATLAB_ZSCORES is True.
        TrialMetadataInvalidationRule(
            condition={"all": [
                {"column": "RT", "op": "<=", "value": MAX_RT_S},
                {"column": "rating", "op": ">=", "value": MIN_RATING},
            ]},
            exclusion_reason="behavioral_threshold",
        ),
        # Delphos vmPFC spike invalidation.
        # EventFileWindowAnnotator(
        #     filter={"suffix": "events", "desc": "delphos"},
        #     metadata_events_key="delphos_events",
        #     window_tmin_s=0.0,
        #     window_tmax_s=5.0,
        # ),
        # EventAnnotationInvalidationRule(
        #     metadata_events_key="delphos_events",
        #     event_filter=build_vmPFC_spike_filter(vm_pfc_channels_by_subject),
        #     exclusion_reason=VM_PFC_SPIKE_EXCLUSION_REASON,
        # ),
    ])

    return annotators


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

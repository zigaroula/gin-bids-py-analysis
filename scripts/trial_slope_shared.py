from __future__ import annotations

import csv
from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np

from gin_bids_py_analysis.bids import BIDSDataset, BIDSFile, BIDSFileGroup, build_subject_groups
from gin_bids_py_analysis.bids.helpers import normalize_subject_value
from gin_bids_py_analysis.processing.hilbert import (
    HilbertParams,
    HilbertWriterParams,
    NormalizationMode,
    ProcessingMethod,
)
from gin_bids_py_analysis.processing.trial_stats.regression import (
    RegressionParams,
    RegressionWriterParams,
)
from gin_bids_py_analysis.processing.trial_stats.result import BaseTrialStatsProcessingResult
from gin_bids_py_analysis.processing.trial_stats_group import (
    RegressionGroupParams,
    RegressionGroupWriterParams,
)
from gin_bids_py_analysis.processing.utils.channels import (
    BipolarDirection,
    BipolarStorage,
    MontageMode,
)
from gin_bids_py_analysis.processing.utils.condition_rules import ConditionExpr
from gin_bids_py_analysis.processing.utils.trial_annotator import (
    DeferredTrialMetadataInvalidationRule,
    EventAnnotationFeatureMaskRule,
    EventAnnotationInvalidationRule,
    EventFileWindowAnnotator,
)
from gin_bids_py_analysis.processing.utils.trial_resolver import ResolvedTrial, TableTrialResolver

# ---------------------------------------------------------------------------
# Trial slope recipe parameters  (edit these)
# ---------------------------------------------------------------------------

BIDS_ROOT = Path(r"E:\Boulot\clarissa_bids")

# Set to a BIDS subject id to restrict the full pipeline, or None for all subjects.
SUBJECT: str | None = None

# Switch the pipeline package here: "regular", "hfo_spike_detection", or "50hz".
# Add new entries to TRIAL_SLOPE_PRESET_OVERRIDES to create more presets.
TRIAL_SLOPE_PRESET: str = "regular"

# One description shared by subject-level regression, group stats, and exports.
TRIAL_SLOPE_OUTPUT_DESCRIPTION = "onset"

# Hilbert derivative used as input for trial-slope stats.
HILBERT_OUTPUT_DESCRIPTION = "bga"
HILBERT_SMOOTHING_WINDOW_MS_FOR_STATS = 250
HILBERT_OUTPUT_FORMAT: Literal["hdf5", "brainvision", "matlab"] = "matlab"

# Pipeline toggles that usually need to stay synchronized across scripts.
ENABLE_HILBERT_NOTCH_FILTER = True
HILBERT_NOTCH_FILTER_FREQS = []  # [50.0]
USE_HFO_SPIKE_EVENT_FILTER = False
HFO_SPIKE_EVENT_FILTER_ROIS = ["vmPFC", "aIns", "daINS", "vaINS"]
HFO_SPIKE_EVENT_FILTER_MODE: Literal["trial", "channel"] = "channel"

# Subject-level trial-slope epoching.
# MATLAB b2 computes trial/channel cleaning masks on the full b1 onset window
# before b3 reads the 0..3.5 s regression interval.  Keep this wide window here
# so the trials/canals sent to regression match the original MATLAB pipeline.
ANCHOR_EVENT_CODES = ["11", "12"]
EXPERIMENT_START_EVENT_CODE = "5"
EXPERIMENT_END_EVENT_CODE: str | None = None
EPOCH_TMIN_S = -1.0
EPOCH_TMAX_S = 6.0

# Optional post-Hilbert notch before epoch extraction. Leave disabled when the
# notch is already applied upstream in Hilbert.
ENABLE_TRIAL_STATS_NOTCH_FILTER = False
TRIAL_STATS_NOTCH_FILTER_FREQS = []

# Execution controls.
HILBERT_N_JOBS = 1
HILBERT_SKIP_EXISTING = True
TRIAL_SLOPE_N_JOBS = 1
TRIAL_SLOPE_SKIP_EXISTING = False
TRIAL_SLOPE_GROUP_N_JOBS = 1

# Export settings for export_trial_slope_group_mean.py.
GROUP_STATS_FILE: Path | None = None
EXPORT_ROI_NAME: str | Sequence[str] = ["aIns", "vmPFC"]
EXPORT_OUTPUT_FORMAT = "png"
EXPORT_OUTPUT_DPI = 300
EXPORT_FIGSIZE_INCHES = (10.4, 6.8)
EXPORT_OUTPUT_DIR = Path("outputs")
EXPORT_OUTPUT_FILE: Path | None = None
EXPORT_TRANSPARENT = False
EXPORT_X_LIMITS: tuple[float, float] | None = [-0.5, 5]
EXPORT_Y_LIMITS: tuple[float, float] | None = None
EXPORT_SHOW_CONTRAST_SIGNIFICANCE_BAR = True
EXPORT_SHOW_VS_ZERO_BOLD_SEGMENTS = True
EXPORT_CONDITION_A_COLOR = "steelblue"
EXPORT_CONDITION_B_COLOR = "tomato"

# ---------------------------------------------------------------------------
# Hilbert configuration
# ---------------------------------------------------------------------------

HILBERT_FILE_FILTERS = {
    "suffix": "ieeg",
    "extension": ".vhdr",
    "task": "valuation"
}

HILBERT_SECONDARY_FILTERS = [
    {
        "scope": "raw",
        "suffix": "events",
        "extension": ".tsv",
        "datatype": "ieeg",
        "task": "valuation"
    },
]

HILBERT_SMOOTHING_WINDOWS_MS = [0, 250, 500, 1000, 2500, 5000]
HILBERT_F_MIN = 50
HILBERT_F_MAX = 150
HILBERT_F_STEP = 10
HILBERT_METHOD = ProcessingMethod.SPM2ENV
HILBERT_COMPUTATION_FREQUENCY_HZ = 512.0
HILBERT_DOWNSAMPLED_FREQUENCY_HZ = 100.0
HILBERT_EVENT_SAMPLE_SHIFT_SAMPLES = 0
HILBERT_EVENTS_SOURCE = "events_tsv"
HILBERT_MONTAGE_MODE = MontageMode.BIPOLAR
HILBERT_BIPOLAR_DIRECTION = BipolarDirection.NEXT_MINUS_PREVIOUS
HILBERT_BIPOLAR_STORAGE = BipolarStorage.NEXT
HILBERT_NORMALIZATION_MODE = NormalizationMode.PERCENT
HILBERT_CHANNELS_FOR_MONTAGE: list[str] | str | None = None
HILBERT_CHANNELS_TO_EXCLUDE_FOR_MONTAGE = (
    r"(?:MKR|DELD|DELG|EOG|ECG|EMG|DC|EXG|EKG|REF|GND|EMPTY).*"
)

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

USE_MATLAB_ZSCORES: bool = True

MATLAB_ZSCORES_PATH = Path(r"E:\data_clarissa\subjects.mat")
MATLAB_ZSCORE_COLUMN_INDEX = 6  # 0-based index for column 7 in trial_characteristics1

# ---------------------------------------------------------------------------
# Trial-slope subject/group configuration
# ---------------------------------------------------------------------------

TRIAL_SLOPE_IEEG_FILTERS = {
    "suffix": "ieeg",
    "extension": ".mat",
    "task": "valuation"
}

TRIAL_SLOPE_SECONDARY_FILTERS = [
    {"scope": "raw", "datatype": "beh", "suffix": "beh", "extension": ".tsv", "task": "valuation"},
    {"scope": "raw", "datatype": "ieeg", "suffix": "events", "extension": ".tsv", "task": "valuation"},
    {"scope": "raw", "datatype": "ieeg", "suffix": "electrodes", "extension": ".tsv"},
    {
        "scope": "hfo_spike_detection",
        "datatype": "ieeg",
        "suffix": "events",
        "extension": ".tsv",
        "desc": "hfospikes",
        "task": "valuation"
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

TRIAL_SLOPE_CONDITION_A = "pleasant"
TRIAL_SLOPE_CONDITION_B = "unpleasant"
TRIAL_SLOPE_EVENTS_SOURCE = "annotations"
TRIAL_SLOPE_PREDICTOR_WITH_MATLAB_ZSCORES = "matlab_zscore"
TRIAL_SLOPE_PREDICTOR_WITHOUT_MATLAB_ZSCORES = "rating"
TRIAL_SLOPE_PREDICTOR_TRANSFORM_BY_CONDITION = {
    "pleasant": {"scale": 1.0, "offset": 0.0},
    "unpleasant": {"scale": -1.0, "offset": 0.0},
}
TRIAL_SLOPE_PREDICTOR_ZSCORE = "none"
TRIAL_SLOPE_ACTIVITY_ZSCORE = "baseline"
TRIAL_SLOPE_ACTIVITY_BASELINE_TMIN_S = -0.25
# MATLAB's nearest-index baseline uses an exclusive upper bound, which
# corresponds to -0.06 s at 100 Hz for the [-0.25, -0.05] request.
TRIAL_SLOPE_ACTIVITY_BASELINE_TMAX_S = -0.06
TRIAL_SLOPE_ACTIVITY_BASELINE_SCOPE = "global"
TRIAL_SLOPE_ACTIVITY_BASELINE_REMOVE_OUTLIER_TRIAL_MEANS = True
TRIAL_SLOPE_ACTIVITY_BASELINE_OUTLIER_METHOD = "median_mad"
TRIAL_SLOPE_P_VALUE_CORRECTION_METHOD = "none"
TRIAL_SLOPE_SIGNIFICANCE_ALPHA = 0.05
TRIAL_SLOPE_ACTIVITY_SUMMARY = {
    "kind": "anchor_to_response_mean",
    "response": {"source": "table_column", "column": "RT", "units": "s"},
}
TRIAL_SLOPE_N_PERMUTATIONS = 0

_PARTICIPANTS_SOURCE_SUBJECT_COLUMNS = (
    "source_subject_id",
    "original_subject_id",
    "original_id",
    "source_id",
)


def _normalize_external_subject_value(subject: object) -> str:
    label = str(subject).strip().replace("_", "")
    return normalize_subject_value(label)


def _subject_lookup_key(subject: object) -> str:
    return _normalize_external_subject_value(subject).casefold()


def load_source_to_bids_subject_map(bids_root: Path = BIDS_ROOT) -> dict[str, str]:
    """Load original subject IDs mapped to BIDS subject labels."""
    participants_path = Path(bids_root) / "participants.tsv"
    if not participants_path.exists():
        return {}

    with participants_path.open("r", encoding="utf-8-sig", newline="") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter="\t")
        if not reader.fieldnames or "participant_id" not in reader.fieldnames:
            return {}

        source_col = next(
            (
                column
                for column in _PARTICIPANTS_SOURCE_SUBJECT_COLUMNS
                if column in reader.fieldnames
            ),
            None,
        )
        if source_col is None:
            return {}

        source_to_bids: dict[str, str] = {}
        for row in reader:
            bids_subject = normalize_subject_value(row.get("participant_id", ""))
            source_subject = _normalize_external_subject_value(row.get(source_col, ""))
            if not bids_subject or not source_subject:
                continue
            source_to_bids[_subject_lookup_key(source_subject)] = bids_subject

        return source_to_bids


def map_source_subject_to_bids(
    subject: object,
    *,
    bids_root: Path = BIDS_ROOT,
    source_to_bids: dict[str, str] | None = None,
) -> str:
    """Return the BIDS subject label for a source/original subject identifier."""
    normalized = _normalize_external_subject_value(subject)
    mapping = (
        load_source_to_bids_subject_map(bids_root)
        if source_to_bids is None
        else source_to_bids
    )
    return mapping.get(_subject_lookup_key(normalized), normalized)


def load_bids_to_source_subject_map(bids_root: Path = BIDS_ROOT) -> dict[str, str]:
    participants_path = Path(bids_root) / "participants.tsv"
    if not participants_path.exists():
        return {}

    with participants_path.open("r", encoding="utf-8-sig", newline="") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter="\t")
        if not reader.fieldnames or "participant_id" not in reader.fieldnames:
            return {}

        source_col = next(
            (
                column
                for column in _PARTICIPANTS_SOURCE_SUBJECT_COLUMNS
                if column in reader.fieldnames
            ),
            None,
        )
        if source_col is None:
            return {}

        bids_to_source: dict[str, str] = {}
        for row in reader:
            bids_subject = normalize_subject_value(row.get("participant_id", ""))
            source_subject = _normalize_external_subject_value(row.get(source_col, ""))
            if not bids_subject or not source_subject:
                continue
            bids_to_source[bids_subject] = source_subject

        return bids_to_source


def map_bids_subject_to_source(
    subject: object,
    *,
    bids_root: Path = BIDS_ROOT,
    bids_to_source: dict[str, str] | None = None,
) -> str:
    """Return the source/original subject label for a BIDS subject label."""
    normalized = normalize_subject_value(str(subject).strip())
    mapping = (
        load_bids_to_source_subject_map(bids_root)
        if bids_to_source is None
        else bids_to_source
    )
    return mapping.get(normalized, _normalize_external_subject_value(normalized))


def event_sample_shift_for_subject(
    subject_id: str,
    *,
    bids_root: Path = BIDS_ROOT,
) -> int:
    """Return the event_sample_shift_samples value for a given BIDS subject.

    gin2bids is configured with different ``event_sample_offset_samples`` per
    recording format so that BIDS event onsets match the SPM ``event.time``
    convention (1-based sample index / sfreq):

    * Micromed (``offset=0``) – BIDS sample = S_trc (1-based).  The Hilbert
      derivative BrainVision annotation is therefore one sample ahead of the
      0-based epoch grid, so a shift of **-1** is required to reproduce the
      MATLAB b1 epoch anchor.
    * Prague (``offset=1``) – BIDS sample = S_raw + 1.  The SPM event.time
      for Prague already incorporates the same +1, so the BIDS annotation sits
      exactly at the intended sample and no shift is needed: **0**.

    Parameters
    ----------
    subject_id : str
        BIDS subject identifier (e.g. ``"01"``).  If ``participants.tsv``
        contains a source subject column, the original identifier is used for
        the Prague/Grenoble format check.
    """
    normalized = map_bids_subject_to_source(subject_id, bids_root=bids_root).upper()
    if normalized.startswith("PRA"):
        return 0
    return -1


ROI_CSV_FILES = {
    "vmPFC": Path(r"E:\Boulot\csv\PFCvm_elecs_tbl.csv"),
    "aIns": Path(r"E:\Boulot\csv\aINS_b5_finite_channels.csv"),
    "daINS": Path(r"E:\Boulot\csv\aINS_dors_elecs_tbl.csv"),
    "vaINS": Path(r"E:\Boulot\csv\aINS_vent_elecs_tbl.csv"),
}

GROUP_ROI_COMBINATIONS = {}
GROUP_KEEP_COMBINED_SOURCE_ROIS = True

GROUP_PARAM_KWARGS = {
    "p_value_correction_method": "cluster_permutation",
    "cluster_permutation_method": "sign_flip",
    "significance_alpha": 0.05,
    "roi_mode": "manual",
    "n_clusters_to_keep": 3,
}

VM_PFC_SPIKE_EXCLUSION_REASON = "spike_or_ripple"
HFO_SPIKE_EVENT_WINDOW_TMIN_S = 0.0
HFO_SPIKE_EVENT_WINDOW_TMAX_S = 6.0

# Behavioral thresholds applied as annotator invalidation rules (orthogonal to
# condition classification).  These replicate MATLAB b2:
#   opts.removeoutlierRTs  → trials with RT > MAX_RT_S are excluded (all channels)
#   opts.removenegratings  → trials with rating < 0 are excluded (all channels)
MAX_RT_S: float = 20.0
MIN_RATING: float = 0.0

REGRESSION_OUTPUT_FORMAT: Literal["hdf5", "matlab"] = "matlab"
REGRESSION_INCLUDE_EPOCHS = False
REGRESSION_GROUP_OUTPUT_FORMAT: Literal["hdf5", "matlab"] = "matlab"

TRIAL_SLOPE_PRESET_OVERRIDES: dict[str, dict[str, Any]] = {
    "regular": {},
    "hfo_spike_detection": {
        "USE_HFO_SPIKE_EVENT_FILTER": True,
        "TRIAL_SLOPE_OUTPUT_DESCRIPTION": "onsethfospikes",
    },
    "50hz": {
        "HILBERT_NOTCH_FILTER_FREQS": [50.0],
        "TRIAL_SLOPE_OUTPUT_DESCRIPTION": "onset50hz",
        "HILBERT_OUTPUT_DESCRIPTION": "bga50hz",
    },
}


def _build_trial_slope_preset_base_settings() -> dict[str, Any]:
    setting_names = {
        name
        for overrides in TRIAL_SLOPE_PRESET_OVERRIDES.values()
        for name in overrides
    }
    unknown_settings = [name for name in setting_names if name not in globals()]
    if unknown_settings:
        raise ValueError(
            "Trial slope preset overrides unknown setting(s): "
            f"{', '.join(sorted(unknown_settings))}."
        )
    return {name: globals()[name] for name in setting_names}


_TRIAL_SLOPE_PRESET_BASE_SETTINGS = _build_trial_slope_preset_base_settings()


def _apply_trial_slope_preset(preset_name: str) -> str:
    preset_key = preset_name.strip().casefold()
    if preset_key not in TRIAL_SLOPE_PRESET_OVERRIDES:
        available_presets = ", ".join(sorted(TRIAL_SLOPE_PRESET_OVERRIDES))
        raise ValueError(
            f"Unknown trial slope preset {preset_name!r}. "
            f"Available presets: {available_presets}."
        )

    for name, value in _TRIAL_SLOPE_PRESET_BASE_SETTINGS.items():
        globals()[name] = value
    for name, value in TRIAL_SLOPE_PRESET_OVERRIDES[preset_key].items():
        globals()[name] = value
    return preset_key


TRIAL_SLOPE_PRESET = _apply_trial_slope_preset(TRIAL_SLOPE_PRESET)

_NA_LIKE_TOKENS = frozenset({"nan", "na", "n/a", "none", "null"})
_FIRST_CONTACT_PATTERN = re.compile(r"^([A-Za-z]+[0-9]+)")


def _output_extension(output_format: str) -> str:
    if output_format == "hdf5":
        return ".h5"
    if output_format == "matlab":
        return ".mat"
    if output_format == "brainvision":
        return ".vhdr"
    raise ValueError(f"Unsupported output format: {output_format}")


@dataclass(frozen=True)
class TrialSlopeRecipe:
    """Single source of truth for the full trial-slope pipeline."""

    preset: str
    bids_root: Path
    subject: str | None
    trial_slope_output_description: str
    hilbert_output_description: str
    hilbert_smoothing_window_ms_for_stats: int
    hilbert_output_format: Literal["hdf5", "brainvision", "matlab"]
    enable_hilbert_notch_filter: bool
    hilbert_notch_filter_freqs: Sequence[float]
    use_hfo_spike_event_filter: bool
    hfo_spike_event_filter_rois: Sequence[str]
    hfo_spike_event_filter_mode: Literal["trial", "channel"]
    anchor_event_codes: Sequence[str]
    experiment_start_event_code: str | None
    experiment_end_event_code: str | None
    epoch_tmin_s: float
    epoch_tmax_s: float
    enable_trial_stats_notch_filter: bool
    trial_stats_notch_filter_freqs: Sequence[float]
    hilbert_file_filters: dict[str, Any]
    hilbert_secondary_filters: list[dict[str, Any]]
    hilbert_smoothing_windows_ms: Sequence[int]
    trial_slope_ieeg_filters: dict[str, Any]
    trial_slope_secondary_filters: list[dict[str, Any]]
    roi_csv_files: dict[str, Path]
    group_roi_combinations: dict[str, Sequence[str]]
    group_keep_combined_source_rois: bool
    group_param_kwargs: dict[str, Any]
    use_matlab_zscores: bool
    matlab_zscores_path: Path
    regression_output_format: Literal["hdf5", "matlab"]
    regression_include_epochs: bool
    regression_group_output_format: Literal["hdf5", "matlab"]

    @property
    def hilbert_derivative_description(self) -> str:
        if self.hilbert_output_format in {"hdf5", "matlab"}:
            return self.hilbert_output_description
        return (
            f"{self.hilbert_output_description}"
            f"sm{self.hilbert_smoothing_window_ms_for_stats}"
        )

    def hilbert_primary_filters(self) -> dict[str, Any]:
        filters = {"scope": "raw", **self.hilbert_file_filters}
        if self.subject:
            filters["subject"] = self.subject
        return filters

    def trial_slope_primary_filters(self) -> dict[str, Any]:
        filters = {
            **self.trial_slope_ieeg_filters,
            "extension": _output_extension(self.hilbert_output_format),
            "desc": self.hilbert_derivative_description,
        }
        if self.subject:
            filters["subject"] = self.subject
        return filters

    def trial_slope_stats_filters(self) -> dict[str, Any]:
        return {
            "scope": "regression",
            "suffix": "stats",
            "extension": _output_extension(self.regression_output_format),
            "desc": self.trial_slope_output_description,
        }

    def group_stats_filters(self) -> dict[str, Any]:
        return {
            "scope": "regression_group",
            "suffix": "stats",
            "extension": _output_extension(self.regression_group_output_format),
            "desc": self.trial_slope_output_description,
        }

    def build_hilbert_params(self) -> HilbertParams:
        notch_freqs = (
            list(self.hilbert_notch_filter_freqs)
            if self.enable_hilbert_notch_filter
            else []
        )
        return HilbertParams(
            f_min=HILBERT_F_MIN,
            f_max=HILBERT_F_MAX,
            f_step=HILBERT_F_STEP,
            method=HILBERT_METHOD,
            computation_frequency_hz=HILBERT_COMPUTATION_FREQUENCY_HZ,
            downsampled_frequency_hz=HILBERT_DOWNSAMPLED_FREQUENCY_HZ,
            notch_filter_freqs=notch_freqs,
            event_sample_shift_samples=HILBERT_EVENT_SAMPLE_SHIFT_SAMPLES,
            events_source=HILBERT_EVENTS_SOURCE,
            smoothing_windows_ms=list(self.hilbert_smoothing_windows_ms),
            montage_mode=HILBERT_MONTAGE_MODE,
            bipolar_direction=HILBERT_BIPOLAR_DIRECTION,
            bipolar_storage=HILBERT_BIPOLAR_STORAGE,
            normalization_mode=HILBERT_NORMALIZATION_MODE,
            channels_for_montage=HILBERT_CHANNELS_FOR_MONTAGE,
            channels_to_exclude_for_montage=HILBERT_CHANNELS_TO_EXCLUDE_FOR_MONTAGE,
        )

    def build_hilbert_writer_params(self) -> HilbertWriterParams:
        return HilbertWriterParams(
            bids_root=self.bids_root,
            output_description=self.hilbert_output_description,
            output_format=self.hilbert_output_format,
        )

    def build_regression_params(self, subject_id: str = "") -> RegressionParams:
        notch_freqs = (
            list(self.trial_stats_notch_filter_freqs)
            if self.enable_trial_stats_notch_filter
            else []
        )
        return RegressionParams(
            anchor_event_codes=list(self.anchor_event_codes),
            # For trial-slope stats the input is already the Hilbert derivative.
            # Its BrainVision annotations carry event times. gin2bids is configured
            # so that BIDS event onsets match the SPM event.time convention (1-based
            # sample index) for all formats (Micromed offset=0, Prague offset=1).
            # The required shift is format-dependent; see event_sample_shift_for_subject.
            events_source=TRIAL_SLOPE_EVENTS_SOURCE,
            event_sample_shift_samples=event_sample_shift_for_subject(
                subject_id,
                bids_root=self.bids_root,
            ),
            experiment_start_event_code=self.experiment_start_event_code,
            experiment_end_event_code=self.experiment_end_event_code,
            tmin_s=self.epoch_tmin_s,
            tmax_s=self.epoch_tmax_s,
            condition_a=TRIAL_SLOPE_CONDITION_A,
            condition_b=TRIAL_SLOPE_CONDITION_B,
            predictor=(
                TRIAL_SLOPE_PREDICTOR_WITH_MATLAB_ZSCORES
                if self.use_matlab_zscores
                else TRIAL_SLOPE_PREDICTOR_WITHOUT_MATLAB_ZSCORES
            ),
            predictor_transform_by_condition=TRIAL_SLOPE_PREDICTOR_TRANSFORM_BY_CONDITION,
            predictor_zscore=TRIAL_SLOPE_PREDICTOR_ZSCORE,
            notch_filter_freqs=notch_freqs,
            activity_zscore=TRIAL_SLOPE_ACTIVITY_ZSCORE,
            activity_baseline_tmin_s=TRIAL_SLOPE_ACTIVITY_BASELINE_TMIN_S,
            activity_baseline_tmax_s=TRIAL_SLOPE_ACTIVITY_BASELINE_TMAX_S,
            activity_baseline_scope=TRIAL_SLOPE_ACTIVITY_BASELINE_SCOPE,
            activity_baseline_remove_outlier_trial_means=(
                TRIAL_SLOPE_ACTIVITY_BASELINE_REMOVE_OUTLIER_TRIAL_MEANS
            ),
            activity_baseline_outlier_method=TRIAL_SLOPE_ACTIVITY_BASELINE_OUTLIER_METHOD,
            p_value_correction_method=TRIAL_SLOPE_P_VALUE_CORRECTION_METHOD,
            significance_alpha=TRIAL_SLOPE_SIGNIFICANCE_ALPHA,
            trial_activity_summary=TRIAL_SLOPE_ACTIVITY_SUMMARY,
            epoch_cleaning=TRIAL_SLOPE_EPOCH_CLEANING,
            n_permutations=TRIAL_SLOPE_N_PERMUTATIONS,
        )

    def build_regression_writer_params(self) -> RegressionWriterParams:
        return RegressionWriterParams(
            bids_root=self.bids_root,
            output_format=self.regression_output_format,
            output_description=self.trial_slope_output_description,
            include_epochs=self.regression_include_epochs,
        )

    def build_regression_group_params(
        self,
        manual_region_channels: dict[str, dict[str, list[str]]] | None = None,
    ) -> RegressionGroupParams:
        channels_by_roi = (
            manual_region_channels
            if manual_region_channels is not None
            else load_roi_channels_from_csv(self.roi_csv_files, bids_root=self.bids_root)
        )
        channels_by_roi = combine_manual_region_channels(
            channels_by_roi,
            self.group_roi_combinations,
            keep_source_rois=self.group_keep_combined_source_rois,
        )
        return RegressionGroupParams(
            manual_region_channels=channels_by_roi,
            **self.group_param_kwargs,
        )

    def build_regression_group_writer_params(self) -> RegressionGroupWriterParams:
        return RegressionGroupWriterParams(
            bids_root=self.bids_root,
            output_format=self.regression_group_output_format,
            output_description=self.trial_slope_output_description,
        )

    def build_trial_slope_groups(self, dataset: BIDSDataset) -> list[BIDSFileGroup]:
        return build_subject_groups(
            dataset,
            self.trial_slope_primary_filters(),
            list(self.trial_slope_secondary_filters),
        )

    def build_hilbert_groups(self, dataset: BIDSDataset) -> list[BIDSFileGroup]:
        return build_subject_groups(
            dataset,
            self.hilbert_primary_filters(),
            list(self.hilbert_secondary_filters),
            aggregate_runs=False,
        )

    def build_trial_annotators(
        self,
        manual_region_channels: dict[str, dict[str, list[str]]],
    ) -> list[Any]:
        hfo_spike_channels_by_subject = build_roi_channels_by_subject(
            manual_region_channels,
            self.hfo_spike_event_filter_rois,
        )
        if self.use_hfo_spike_event_filter and not hfo_spike_channels_by_subject:
            roi_label = ", ".join(self.hfo_spike_event_filter_rois) or "<none>"
            raise ValueError(
                "At least one ROI with channels is required to exclude HFO/spike "
                f"event trials. Requested ROI(s): {roi_label}."
            )

        annotators: list[Any] = []

        if self.use_matlab_zscores:
            annotators.append(
                MatlabZscorePredictorAnnotator(
                    self.matlab_zscores_path,
                    bids_root=self.bids_root,
                )
            )

        annotators.extend([
            # Behavioral validity is deferred to mirror MATLAB b2 ordering:
            # RT outliers are NaN'd after HGA trial-level mean/max rejection but before
            # channel-level spread/25%-NaN checks; negative ratings are NaN'd after
            # channel-level cleaning and before baseline z-scoring/regression.
            # NOTE: These filters still apply to RAW rating values even when
            # USE_MATLAB_ZSCORES is True.
            DeferredTrialMetadataInvalidationRule(
                condition={"column": "RT", "op": "<=", "value": MAX_RT_S},
                exclusion_reason="outlier_rt",
                apply_phase="after_epoch_trial_rejection",
            ),
            DeferredTrialMetadataInvalidationRule(
                condition={"column": "rating", "op": ">=", "value": MIN_RATING},
                exclusion_reason="negative_rating",
                apply_phase="before_activity_zscore",
            ),
        ])

        if self.use_hfo_spike_event_filter:
            hfo_spike_event_filter = build_roi_spike_filter(hfo_spike_channels_by_subject)
            if self.hfo_spike_event_filter_mode == "trial":
                hfo_spike_filter_rule: Any = EventAnnotationInvalidationRule(
                    metadata_events_key="hfo_spike_events",
                    event_filter=hfo_spike_event_filter,
                    exclusion_reason=VM_PFC_SPIKE_EXCLUSION_REASON,
                )
            elif self.hfo_spike_event_filter_mode == "channel":
                hfo_spike_filter_rule = EventAnnotationFeatureMaskRule(
                    metadata_events_key="hfo_spike_events",
                    event_filter=hfo_spike_event_filter,
                    exclusion_reason=VM_PFC_SPIKE_EXCLUSION_REASON,
                )
            else:
                raise ValueError(
                    "Unsupported HFO/spike event filter mode: "
                    f"{self.hfo_spike_event_filter_mode!r}. Expected 'trial' or 'channel'."
                )
            annotators.extend([
                EventFileWindowAnnotator(
                    filter={"suffix": "events", "desc": "hfospikes"},
                    metadata_events_key="hfo_spike_events",
                    window_tmin_s=HFO_SPIKE_EVENT_WINDOW_TMIN_S,
                    window_tmax_s=HFO_SPIKE_EVENT_WINDOW_TMAX_S,
                ),
                hfo_spike_filter_rule,
            ])

        return annotators

    def summary_lines(self) -> list[str]:
        return [
            f"Preset: {self.preset}",
            f"BIDS root: {self.bids_root}",
            f"subject filter: {self.subject or 'all'}",
            f"Hilbert desc: {self.hilbert_output_description}",
            f"Trial-slope input desc: {self.hilbert_derivative_description}",
            f"Stats/group desc: {self.trial_slope_output_description}",
            f"Hilbert output format: {self.hilbert_output_format}",
            (
                "Hilbert notch: "
                f"{list(self.hilbert_notch_filter_freqs) if self.enable_hilbert_notch_filter else 'off'}"
            ),
            f"HFO/spike event filter: {self.use_hfo_spike_event_filter}",
            f"HFO/spike event ROI(s): {list(self.hfo_spike_event_filter_rois)}",
            f"HFO/spike event mode: {self.hfo_spike_event_filter_mode}",
            f"Group ROI combinations: {self.group_roi_combinations or 'none'}",
            (
                "Keep combined source ROI(s): "
                f"{self.group_keep_combined_source_rois}"
            ),
            (
                "Epoch: "
                f"anchors={list(self.anchor_event_codes)}, "
                f"window=[{self.epoch_tmin_s}, {self.epoch_tmax_s}], "
                f"start={self.experiment_start_event_code}, "
                f"end={self.experiment_end_event_code}"
            ),
        ]


RECIPE = TrialSlopeRecipe(
    preset=TRIAL_SLOPE_PRESET,
    bids_root=BIDS_ROOT,
    subject=SUBJECT,
    trial_slope_output_description=TRIAL_SLOPE_OUTPUT_DESCRIPTION,
    hilbert_output_description=HILBERT_OUTPUT_DESCRIPTION,
    hilbert_smoothing_window_ms_for_stats=HILBERT_SMOOTHING_WINDOW_MS_FOR_STATS,
    hilbert_output_format=HILBERT_OUTPUT_FORMAT,
    enable_hilbert_notch_filter=ENABLE_HILBERT_NOTCH_FILTER,
    hilbert_notch_filter_freqs=HILBERT_NOTCH_FILTER_FREQS,
    use_hfo_spike_event_filter=USE_HFO_SPIKE_EVENT_FILTER,
    hfo_spike_event_filter_rois=HFO_SPIKE_EVENT_FILTER_ROIS,
    hfo_spike_event_filter_mode=HFO_SPIKE_EVENT_FILTER_MODE,
    anchor_event_codes=ANCHOR_EVENT_CODES,
    experiment_start_event_code=EXPERIMENT_START_EVENT_CODE,
    experiment_end_event_code=EXPERIMENT_END_EVENT_CODE,
    epoch_tmin_s=EPOCH_TMIN_S,
    epoch_tmax_s=EPOCH_TMAX_S,
    enable_trial_stats_notch_filter=ENABLE_TRIAL_STATS_NOTCH_FILTER,
    trial_stats_notch_filter_freqs=TRIAL_STATS_NOTCH_FILTER_FREQS,
    hilbert_file_filters=HILBERT_FILE_FILTERS,
    hilbert_secondary_filters=HILBERT_SECONDARY_FILTERS,
    hilbert_smoothing_windows_ms=HILBERT_SMOOTHING_WINDOWS_MS,
    trial_slope_ieeg_filters=TRIAL_SLOPE_IEEG_FILTERS,
    trial_slope_secondary_filters=TRIAL_SLOPE_SECONDARY_FILTERS,
    roi_csv_files=ROI_CSV_FILES,
    group_roi_combinations=GROUP_ROI_COMBINATIONS,
    group_keep_combined_source_rois=GROUP_KEEP_COMBINED_SOURCE_ROIS,
    group_param_kwargs=GROUP_PARAM_KWARGS,
    use_matlab_zscores=USE_MATLAB_ZSCORES,
    matlab_zscores_path=MATLAB_ZSCORES_PATH,
    regression_output_format=REGRESSION_OUTPUT_FORMAT,
    regression_include_epochs=REGRESSION_INCLUDE_EPOCHS,
    regression_group_output_format=REGRESSION_GROUP_OUTPUT_FORMAT,
)

# Backward-compatible aliases used by the comparison and visualization scripts.
IEEG_FILTERS = RECIPE.trial_slope_primary_filters()
SECONDARY_FILTERS = list(TRIAL_SLOPE_SECONDARY_FILTERS)
TRIAL_SLOPE_STATS_FILTERS = RECIPE.trial_slope_stats_filters()
GROUP_STATS_FILTERS = RECIPE.group_stats_filters()


def build_params(subject_id: str = "") -> RegressionParams:
    """Build subject-level regression params for the current recipe."""
    return RECIPE.build_regression_params(subject_id)


def build_hilbert_params() -> HilbertParams:
    return RECIPE.build_hilbert_params()


def build_hilbert_writer_params() -> HilbertWriterParams:
    return RECIPE.build_hilbert_writer_params()


def build_regression_writer_params() -> RegressionWriterParams:
    return RECIPE.build_regression_writer_params()


def build_regression_group_writer_params() -> RegressionGroupWriterParams:
    return RECIPE.build_regression_group_writer_params()


def print_recipe_summary(recipe: TrialSlopeRecipe = RECIPE) -> None:
    print("Trial slope recipe:")
    for line in recipe.summary_lines():
        print(f"  - {line}")


PARAMS = build_params()


def build_trial_slope_resolver() -> TableTrialResolver:
    return TableTrialResolver(
        conditions=TRIAL_SLOPE_RESOLVER_CONDITIONS,
        extract_columns=TRIAL_SLOPE_EXTRACT_COLUMNS,
        filter={"suffix": "beh"},
    )


RESOLVER = build_trial_slope_resolver()


def _is_nan_like(value: object) -> bool:
    return str(value).strip().casefold() in _NA_LIKE_TOKENS


def _normalize_subject_from_csv(
    raw_subject: object,
    *,
    source_to_bids: dict[str, str],
) -> str:
    return map_source_subject_to_bids(raw_subject, source_to_bids=source_to_bids)


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
    *,
    bids_root: Path = BIDS_ROOT,
) -> dict[str, dict[str, list[str]]]:
    manual_region_channels: dict[str, dict[str, list[str]]] = {}
    source_to_bids = load_source_to_bids_subject_map(bids_root)

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

                subject = _normalize_subject_from_csv(
                    row.get(subject_col, ""),
                    source_to_bids=source_to_bids,
                )
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


def combine_manual_region_channels(
    manual_region_channels: dict[str, dict[str, list[str]]],
    roi_combinations: dict[str, Sequence[str]],
    *,
    keep_source_rois: bool = False,
) -> dict[str, dict[str, list[str]]]:
    if not roi_combinations:
        return {
            roi: {subject: list(channels) for subject, channels in subject_map.items()}
            for roi, subject_map in manual_region_channels.items()
        }

    combined: dict[str, dict[str, list[str]]] = {
        roi: {subject: list(channels) for subject, channels in subject_map.items()}
        for roi, subject_map in manual_region_channels.items()
    }
    source_rois_to_remove: set[str] = set()

    for raw_target_roi, raw_source_rois in roi_combinations.items():
        target_roi = str(raw_target_roi).strip()
        if not target_roi:
            raise ValueError("Combined ROI names cannot be empty.")

        source_rois = [str(roi).strip() for roi in raw_source_rois if str(roi).strip()]
        if not source_rois:
            raise ValueError(
                f"Combined ROI {target_roi!r} requires at least one source ROI."
            )

        missing_source_rois = [
            roi for roi in source_rois if roi not in manual_region_channels
        ]
        if missing_source_rois:
            raise ValueError(
                f"Combined ROI {target_roi!r} references missing source ROI(s): "
                f"{', '.join(missing_source_rois)}."
            )

        target_subject_map: dict[str, list[str]] = {}
        seen_by_subject: dict[str, set[str]] = {}
        for source_roi in source_rois:
            source_rois_to_remove.add(source_roi)
            for subject, channels in manual_region_channels[source_roi].items():
                subject_key = str(subject).strip()
                if not subject_key:
                    continue
                seen = seen_by_subject.setdefault(subject_key, set())
                for channel in channels:
                    cleaned = str(channel).strip()
                    if not cleaned:
                        continue
                    channel_key = cleaned.casefold()
                    if channel_key in seen:
                        continue
                    seen.add(channel_key)
                    target_subject_map.setdefault(subject_key, []).append(cleaned)

        if target_subject_map:
            combined[target_roi] = target_subject_map

    if not keep_source_rois:
        for source_roi in source_rois_to_remove:
            if source_roi not in roi_combinations:
                combined.pop(source_roi, None)

    return combined


def print_roi_summary(manual_region_channels: dict[str, dict[str, list[str]]]) -> None:
    for roi_name, subject_map in manual_region_channels.items():
        n_subjects = len(subject_map)
        n_channels = sum(len(channels) for channels in subject_map.values())
        print(f"ROI {roi_name}: {n_channels} channel(s) across {n_subjects} subject(s).")


def build_roi_channels_by_subject(
    manual_region_channels: dict[str, dict[str, list[str]]],
    roi_names: Sequence[str],
) -> dict[str, list[str]]:
    channels_by_subject: dict[str, list[str]] = {}
    seen_by_subject: dict[str, set[str]] = {}

    for raw_roi_name in roi_names:
        roi_name = str(raw_roi_name).strip()
        if not roi_name:
            continue
        subject_channels = manual_region_channels.get(roi_name, {})
        for subject_id, channels in subject_channels.items():
            subject = str(subject_id).strip()
            if not subject:
                continue
            seen = seen_by_subject.setdefault(subject, set())
            for channel in channels:
                cleaned = str(channel).strip()
                if not cleaned:
                    continue
                channel_key = cleaned.casefold()
                if channel_key in seen:
                    continue
                seen.add(channel_key)
                channels_by_subject.setdefault(subject, []).append(cleaned)

    return channels_by_subject


def build_roi_spike_filter(
    roi_channels_by_subject: dict[str, list[str]],
) -> ConditionExpr:
    per_subject_filters: list[ConditionExpr] = []

    for subject_id, channels in sorted(roi_channels_by_subject.items()):
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
                    ConditionExpr(
                        any=[
                            ConditionExpr(column="event_type", op="==", value="Spike"),
                            ConditionExpr(column="event_type", op="==", value="Ripple"),
                        ]
                    ),
                    ConditionExpr(column="channel", op="in", values=unique_channels),
                ]
            )
        )

    if not per_subject_filters:
        raise ValueError("No ROI channels available to build the HFO/spike event filter.")

    if len(per_subject_filters) == 1:
        return per_subject_filters[0]

    return ConditionExpr(any=per_subject_filters)


def build_vmPFC_spike_filter(
    vm_pfc_channels_by_subject: dict[str, list[str]],
) -> ConditionExpr:
    """Backward-compatible alias for the historical vmPFC-only filter."""
    return build_roi_spike_filter(vm_pfc_channels_by_subject)


def load_matlab_zscores(
    mat_path: Path,
    *,
    bids_root: Path = BIDS_ROOT,
) -> dict[str, np.ndarray]:
    """Load z-scored predictor values from MATLAB subjects.mat file.

    Parameters
    ----------
    mat_path : Path
        Path to the MATLAB .mat file containing subject trial characteristics.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary mapping BIDS subject IDs to z-score arrays.  If
        ``participants.tsv`` contains source/original subject IDs, MATLAB names
        are translated through that table.

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

    source_to_bids = load_source_to_bids_subject_map(bids_root)
    zscores_by_subject: dict[str, np.ndarray] = {}

    for subject_obj in subjects_data:
        if not hasattr(subject_obj, "name"):
            continue

        raw_name = str(subject_obj.name).strip()
        if not raw_name:
            continue

        bids_subject = map_source_subject_to_bids(
            raw_name,
            source_to_bids=source_to_bids,
        )

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
        zscores_by_subject[bids_subject] = zscores

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

    def __init__(self, mat_path: Path, *, bids_root: Path = BIDS_ROOT) -> None:
        self.zscores_by_subject = load_matlab_zscores(mat_path, bids_root=bids_root)

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

        zscores = self.zscores_by_subject.get(normalize_subject_value(subject_id))
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
) -> list[Any]:
    """Build the list of trial annotators for trial slope processing.

    When USE_MATLAB_ZSCORES is True, a MatlabZscorePredictorAnnotator is prepended
    to inject z-scored predictor values before behavioral filtering.

    Parameters
    ----------
    manual_region_channels : dict[str, dict[str, list[str]]]
        ROI channel mapping by subject, used for optional HFO/spike event exclusion.

    Returns
    -------
    list
        List of annotators including optional MATLAB z-score injection,
        behavioral thresholds, and HFO/spike event filtering.

    Raises
    ------
    ValueError
        If the HFO/spike event filter is enabled and no selected ROI channels are provided.
    """
    return RECIPE.build_trial_annotators(manual_region_channels)


def build_group_params(
    manual_region_channels: dict[str, dict[str, list[str]]] | None = None,
) -> RegressionGroupParams:
    return RECIPE.build_regression_group_params(manual_region_channels)


def build_hilbert_groups(
    dataset: BIDSDataset,
) -> list[BIDSFileGroup]:
    return RECIPE.build_hilbert_groups(dataset)


def build_trial_slope_groups(
    dataset: BIDSDataset,
) -> list[BIDSFileGroup]:
    return RECIPE.build_trial_slope_groups(dataset)


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
        1
        for trial in result.resolved_trials
        if getattr(trial, "metadata", {}).get("nan_masked_features")
    )
    dropped_channels = len(
        getattr(result, "epoch_cleaning_audit", {}).get("excluded_features", {})
    )
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


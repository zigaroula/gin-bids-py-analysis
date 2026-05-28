"""Processor for per-subject slope regression (gamma ~ predictor) on iEEG derivatives."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from mne.io import BaseRaw

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.processing.utils.statistics import compute_condition_mean, compute_condition_sem
from bidsforge.processing.utils.trial_annotator import TrialWindowAnnotator
from bidsforge.processing.utils.trial_resolver import ResolvedTrial, TrialResolver

from ..processor import BaseTrialStatsProcessing, TrialStatsProcessingContext
from ..result import ConditionSignalActivity, SignalActivityEstimate
from .params import RegressionParams
from .result import (
    ConditionPredictorValues,
    ConditionRegressionStats,
    RegressionPredictor,
    RegressionProcessingResult,
    RegressionStats,
)
from .stats import (
    compute_linear_regression_maps,
    compute_permuted_regression_maps,
    zscore_epochs_across_trials,
    zscore_predictor_values_by_scope,
)


class RegressionProcessing(BaseTrialStatsProcessing):
    """Compute per-subject per-condition slope regression on epoched iEEG data."""

    def __init__(
        self,
        params: RegressionParams,
        resolver: TrialResolver,
        annotators: Sequence[TrialWindowAnnotator] = (),
    ) -> None:
        super().__init__(params=params, resolver=resolver, annotators=annotators)

    @property
    def params(self) -> RegressionParams:  # type: ignore[override]
        return self._params

    @params.setter
    def params(self, value: RegressionParams) -> None:
        self._params = value

    def _initialize_pipeline_state(self) -> dict[str, object]:
        state = super()._initialize_pipeline_state()
        state.update(
            {
            "excluded_channels": {},
            "excluded_trial_channel_pairs": {},
            "activity_a_zscore_valid": True,
            "activity_b_zscore_valid": True,
            }
        )
        return state

    def _normalize_trials(
        self,
        *,
        group: BIDSFileGroup,
        ieeg_file: BIDSFile,
        raw: BaseRaw,
        anchor_events: list[object],
        trials: list[ResolvedTrial],
    ) -> list[ResolvedTrial]:
        del group, ieeg_file, raw, anchor_events
        normalized: list[ResolvedTrial] = []
        supported_labels = {self.params.condition_a, self.params.condition_b}
        predictor_key = self.params.predictor
        for trial in trials:
            metadata = dict(trial.metadata)
            raw_predictor = metadata.get(predictor_key)
            metadata["predictor_raw"] = "" if raw_predictor is None else str(raw_predictor)
            raw_predictor_value = _to_float_or_nan(raw_predictor)
            metadata["predictor_raw_value"] = (
                float(raw_predictor_value) if np.isfinite(raw_predictor_value) else np.nan
            )
            transform = self.params.predictor_transform_by_condition.get(
                trial.label,
                self.params.predictor_transform_by_condition.get(str(trial.label).strip()),
            )
            scale = float(transform.scale) if transform is not None else 1.0
            offset = float(transform.offset) if transform is not None else 0.0
            metadata["predictor_transform_scale"] = scale
            metadata["predictor_transform_offset"] = offset
            predictor_value = _apply_affine_transform(
                raw_predictor_value,
                scale=scale,
                offset=offset,
            )
            metadata["predictor_transformed_value"] = (
                float(predictor_value) if np.isfinite(predictor_value) else np.nan
            )
            metadata["predictor_value"] = (
                float(predictor_value) if np.isfinite(predictor_value) else np.nan
            )

            if not trial.keep:
                normalized.append(replace(trial, metadata=metadata))
                continue
            if trial.label not in supported_labels:
                normalized.append(
                    ResolvedTrial(
                        source_file=trial.source_file,
                        anchor_event_index=trial.anchor_event_index,
                        anchor_event_code=trial.anchor_event_code,
                        anchor_onset_s=trial.anchor_onset_s,
                        anchor_duration_s=trial.anchor_duration_s,
                        label=trial.label,
                        trial_id=trial.trial_id,
                        keep=False,
                        exclusion_reason=trial.exclusion_reason or "unsupported_label",
                        metadata=metadata,
                    )
                )
                continue
            if not np.isfinite(predictor_value):
                normalized.append(
                    ResolvedTrial(
                        source_file=trial.source_file,
                        anchor_event_index=trial.anchor_event_index,
                        anchor_event_code=trial.anchor_event_code,
                        anchor_onset_s=trial.anchor_onset_s,
                        anchor_duration_s=trial.anchor_duration_s,
                        label=trial.label,
                        trial_id=trial.trial_id,
                        keep=False,
                        exclusion_reason=trial.exclusion_reason or "invalid_predictor_value",
                        metadata=metadata,
                    )
                )
                continue
            normalized.append(replace(trial, metadata=metadata))

        return normalized

    def _prepare_epochs_before_activity_zscore(
        self,
        *,
        epochs_a: np.ndarray,
        epochs_b: np.ndarray,
        kept_trials_a: list[ResolvedTrial],
        kept_trials_b: list[ResolvedTrial],
        feature_names: list[str],
        feature_indices: list[np.ndarray] | None,
        time_axis_s: np.ndarray,
        state: dict[str, object],
        atlas_mode: bool,
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[np.ndarray] | None]:
        epochs_a, epochs_b, feature_names, feature_indices = super()._prepare_epochs_before_activity_zscore(
            epochs_a=epochs_a,
            epochs_b=epochs_b,
            kept_trials_a=kept_trials_a,
            kept_trials_b=kept_trials_b,
            feature_names=feature_names,
            feature_indices=feature_indices,
            time_axis_s=time_axis_s,
            state=state,
            atlas_mode=atlas_mode,
        )
        audit = dict(state.get("epoch_cleaning_audit", {}))
        excluded_features = dict(audit.get("excluded_features", {}))
        nan_masked_pairs = {
            str(name): [int(idx) for idx in indices]
            for name, indices in dict(audit.get("nan_masked_trial_feature_pairs", {})).items()
            if str(name) not in excluded_features
        }
        state["excluded_channels"] = excluded_features
        state["excluded_trial_channel_pairs"] = nan_masked_pairs
        return epochs_a, epochs_b, feature_names, feature_indices

    def _apply_activity_zscore(
        self,
        *,
        epochs_a: np.ndarray,
        epochs_b: np.ndarray,
        time_axis_s: np.ndarray,
        state: dict[str, object],
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.params.activity_zscore == "baseline":
            epochs_a, epochs_b = super()._apply_activity_zscore(
                epochs_a=epochs_a,
                epochs_b=epochs_b,
                time_axis_s=time_axis_s,
                state=state,
            )
            state["activity_a_zscore_valid"] = True
            state["activity_b_zscore_valid"] = True
            return epochs_a, epochs_b
        if self.params.activity_zscore == "across_trials":
            epochs_a, valid_a = zscore_epochs_across_trials(epochs_a)
            epochs_b, valid_b = zscore_epochs_across_trials(epochs_b)
            state["activity_a_zscore_valid"] = valid_a
            state["activity_b_zscore_valid"] = valid_b
            return epochs_a, epochs_b
        state["activity_a_zscore_valid"] = True
        state["activity_b_zscore_valid"] = True
        return epochs_a, epochs_b

    def _build_pipeline_metadata(self, *, state: dict[str, object]) -> dict[str, object]:
        del state
        return {
            "analysis_type": "slope_regression",
            "predictor": self.params.predictor,
            "predictor_zscore": self.params.predictor_zscore,
            "predictor_transform_by_condition": {
                condition: {
                    "scale": float(transform.scale),
                    "offset": float(transform.offset),
                }
                for condition, transform in self.params.predictor_transform_by_condition.items()
            },
        }

    def _compute_and_build_result(
        self,
        context: TrialStatsProcessingContext,
    ) -> RegressionProcessingResult:
        kept_trials_a = list(context.kept_trials_a)
        kept_trials_b = list(context.kept_trials_b)
        predictor_a_raw_array = _trial_metadata_values(kept_trials_a, "predictor_raw_value")
        predictor_b_raw_array = _trial_metadata_values(kept_trials_b, "predictor_raw_value")
        predictor_a_transformed_array = _trial_metadata_values(
            kept_trials_a,
            "predictor_transformed_value",
        )
        predictor_b_transformed_array = _trial_metadata_values(
            kept_trials_b,
            "predictor_transformed_value",
        )

        n_features = len(context.feature_names)
        n_times = len(context.time_axis_eval)
        condition_a_mean = compute_condition_mean(
            context.epochs_a,
            n_features=n_features,
            n_times=n_times,
        )
        condition_b_mean = compute_condition_mean(
            context.epochs_b,
            n_features=n_features,
            n_times=n_times,
        )
        condition_a_sem = compute_condition_sem(
            context.epochs_a,
            n_features=n_features,
            n_times=n_times,
        )
        condition_b_sem = compute_condition_sem(
            context.epochs_b,
            n_features=n_features,
            n_times=n_times,
        )

        cond_a_ready = (
            context.epochs_a.shape[0] >= self.params.min_trials_per_condition
            and bool(context.state.get("activity_a_zscore_valid", True))
        )
        cond_b_ready = (
            context.epochs_b.shape[0] >= self.params.min_trials_per_condition
            and bool(context.state.get("activity_b_zscore_valid", True))
        )

        (
            predictor_a_effective_array,
            predictor_b_effective_array,
            predictor_a_zscore_valid,
            predictor_b_zscore_valid,
        ) = self._scale_predictor_values_pair(
            predictor_a_transformed_array,
            predictor_b_transformed_array,
        )
        self._store_effective_predictor_values(kept_trials_a, predictor_a_effective_array)
        self._store_effective_predictor_values(kept_trials_b, predictor_b_effective_array)

        (
            condition_a_slope,
            condition_a_intercept,
            condition_a_r_value,
            condition_a_p_value,
            condition_a_stats_valid,
        ) = compute_linear_regression_maps(
            predictor_a_effective_array
            if cond_a_ready and predictor_a_zscore_valid
            else np.array([], dtype=np.float64),
            context.epochs_a if cond_a_ready else np.empty_like(context.epochs_a[:0]),
            n_features=n_features,
            n_times=n_times,
        )
        (
            condition_b_slope,
            condition_b_intercept,
            condition_b_r_value,
            condition_b_p_value,
            condition_b_stats_valid,
        ) = compute_linear_regression_maps(
            predictor_b_effective_array
            if cond_b_ready and predictor_b_zscore_valid
            else np.array([], dtype=np.float64),
            context.epochs_b if cond_b_ready else np.empty_like(context.epochs_b[:0]),
            n_features=n_features,
            n_times=n_times,
        )

        from bidsforge.processing.utils.statistics import correct_p_values

        condition_a_p_value_corrected = correct_p_values(
            condition_a_p_value,
            method=self.params.p_value_correction_method,
        )
        condition_b_p_value_corrected = correct_p_values(
            condition_b_p_value,
            method=self.params.p_value_correction_method,
        )
        condition_a_significant_mask = np.isfinite(condition_a_p_value_corrected) & (
            condition_a_p_value_corrected < self.params.significance_alpha
        )
        condition_b_significant_mask = np.isfinite(condition_b_p_value_corrected) & (
            condition_b_p_value_corrected < self.params.significance_alpha
        )

        n_perms = self.params.n_permutations
        condition_a_permuted_slopes: np.ndarray | None = None
        condition_b_permuted_slopes: np.ndarray | None = None
        if n_perms > 0:
            rng = np.random.default_rng(self.params.permutation_seed)
            if condition_a_stats_valid:
                condition_a_permuted_slopes = compute_permuted_regression_maps(
                    predictor_a_effective_array,
                    context.epochs_a,
                    n_perm=n_perms,
                    rng=rng,
                    n_features=n_features,
                    n_times=n_times,
                )
            if condition_b_stats_valid:
                condition_b_permuted_slopes = compute_permuted_regression_maps(
                    predictor_b_effective_array,
                    context.epochs_b,
                    n_perm=n_perms,
                    rng=rng,
                    n_features=n_features,
                    n_times=n_times,
                )

        return RegressionProcessingResult(
            **self._build_common_result_kwargs(context),
            regression=RegressionStats(
                condition_a=ConditionRegressionStats(
                    slope=condition_a_slope,
                    intercept=condition_a_intercept,
                    r_value=condition_a_r_value,
                    p_value=condition_a_p_value,
                    p_value_corrected=condition_a_p_value_corrected,
                    significant_mask=condition_a_significant_mask,
                    n_trials_used=int(context.epochs_a.shape[0]),
                    stats_valid=condition_a_stats_valid,
                    permuted_slopes=condition_a_permuted_slopes,
                ),
                condition_b=ConditionRegressionStats(
                    slope=condition_b_slope,
                    intercept=condition_b_intercept,
                    r_value=condition_b_r_value,
                    p_value=condition_b_p_value,
                    p_value_corrected=condition_b_p_value_corrected,
                    significant_mask=condition_b_significant_mask,
                    n_trials_used=int(context.epochs_b.shape[0]),
                    stats_valid=condition_b_stats_valid,
                    permuted_slopes=condition_b_permuted_slopes,
                ),
            ),
            predictor_values=RegressionPredictor(
                condition_a=ConditionPredictorValues(
                    raw_values=predictor_a_raw_array,
                    transformed_values=predictor_a_transformed_array,
                    values=predictor_a_effective_array,
                ),
                condition_b=ConditionPredictorValues(
                    raw_values=predictor_b_raw_array,
                    transformed_values=predictor_b_transformed_array,
                    values=predictor_b_effective_array,
                ),
            ),
            signal_activity=ConditionSignalActivity(
                condition_a=SignalActivityEstimate(mean=condition_a_mean, sem=condition_a_sem),
                condition_b=SignalActivityEstimate(mean=condition_b_mean, sem=condition_b_sem),
            ),
            analysis_type="slope_regression",
            excluded_channels=dict(context.state["excluded_channels"]),
            excluded_trial_channel_pairs=dict(context.state["excluded_trial_channel_pairs"]),
            predictor=self.params.predictor,
            predictor_zscore=self.params.predictor_zscore,
            predictor_transform_by_condition={
                condition: {
                    "scale": float(transform.scale),
                    "offset": float(transform.offset),
                }
                for condition, transform in self.params.predictor_transform_by_condition.items()
            },
            stats_valid=bool(condition_a_stats_valid or condition_b_stats_valid),
        )

    def _scale_predictor_values_pair(
        self,
        predictor_values_a: np.ndarray,
        predictor_values_b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        return zscore_predictor_values_by_scope(
            predictor_values_a,
            predictor_values_b,
            scope=self.params.predictor_zscore,
        )

    def _store_effective_predictor_values(
        self,
        trials: list[ResolvedTrial],
        predictor_values: np.ndarray,
    ) -> None:
        for trial, value in zip(trials, np.asarray(predictor_values, dtype=np.float64).ravel()):
            trial.metadata["predictor_value"] = float(value) if np.isfinite(value) else np.nan


def _to_float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, (int, float, np.integer, np.floating)):
        out = float(value)
        return out if np.isfinite(out) else float("nan")
    text = str(value).strip()
    if not text:
        return float("nan")
    try:
        out = float(text)
    except ValueError:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _apply_affine_transform(
    value: float,
    *,
    scale: float,
    offset: float,
) -> float:
    if not np.isfinite(value):
        return float("nan")
    transformed = (float(scale) * float(value)) + float(offset)
    return transformed if np.isfinite(transformed) else float("nan")


def _trial_metadata_values(
    trials: list[ResolvedTrial],
    key: str,
) -> np.ndarray:
    return np.asarray(
        [_to_float_or_nan(trial.metadata.get(key)) for trial in trials],
        dtype=np.float64,
    )

"""Processor for per-subject slope regression (gamma ~ predictor) on iEEG derivatives."""

from __future__ import annotations

from dataclasses import replace
import json

import numpy as np
from mne.io import BaseRaw

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.utils.epoch_quality import (
    apply_channel_exclusions,
    apply_trial_nan_mask,
    detect_outlier_trial_channel_pairs_by_max,
    detect_outlier_trial_channel_pairs_by_mean,
    reject_channels_by_nan_trial_ratio,
    reject_channels_by_trial_max_spread,
    reject_channels_by_trial_mean_spread,
)
from gin_bids_py_analysis.processing.utils.statistics import compute_condition_mean, compute_condition_sem
from gin_bids_py_analysis.processing.utils.trial_resolver import ResolvedTrial, TrialResolver

from ..processor import BaseTrialStatsProcessing, TrialStatsProcessingContext
from .params import RegressionParams
from .result import RegressionProcessingResult
from .stats import (
    compute_linear_regression_maps,
    zscore_epochs_across_trials,
    zscore_predictor_values_by_scope,
)


class RegressionProcessing(BaseTrialStatsProcessing):
    """Compute per-subject per-condition slope regression on epoched iEEG data."""

    def __init__(
        self,
        params: RegressionParams,
        resolver: TrialResolver,
    ) -> None:
        super().__init__(params=params, resolver=resolver)

    @property
    def params(self) -> RegressionParams:  # type: ignore[override]
        return self._params

    @params.setter
    def params(self, value: RegressionParams) -> None:
        self._params = value

    def _initialize_pipeline_state(self) -> dict[str, object]:
        return {
            "predictor_a_raw": [],
            "predictor_b_raw": [],
            "predictor_a_transformed": [],
            "predictor_b_transformed": [],
            "excluded_channels": {},
            "excluded_trial_channel_pairs": {},
            "activity_a_zscore_valid": True,
            "activity_b_zscore_valid": True,
        }

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

    def _on_kept_trial_epoch(
        self,
        *,
        epoch_for_stats: np.ndarray,
        trial: ResolvedTrial,
        condition: str,
        state: dict[str, object],
    ) -> None:
        del epoch_for_stats
        if condition == "a":
            state["predictor_a_raw"].append(_to_float_or_nan(trial.metadata.get("predictor_raw_value")))
            state["predictor_a_transformed"].append(
                _to_float_or_nan(trial.metadata.get("predictor_transformed_value"))
            )
        else:
            state["predictor_b_raw"].append(_to_float_or_nan(trial.metadata.get("predictor_raw_value")))
            state["predictor_b_transformed"].append(
                _to_float_or_nan(trial.metadata.get("predictor_transformed_value"))
            )

    def _prepare_epochs_before_activity_zscore(
        self,
        *,
        epochs_a: np.ndarray,
        epochs_b: np.ndarray,
        feature_names: list[str],
        feature_indices: list[np.ndarray] | None,
        time_axis_s: np.ndarray,
        state: dict[str, object],
        atlas_mode: bool,
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[np.ndarray] | None]:
        del time_axis_s
        excluded_channels: dict[str, str] = {}
        excluded_trial_channel_pairs: dict[str, list[int]] = {}

        cfg = self.params.epoch_cleaning
        any_level_a = cfg.reject_trials_by_epoch_mean or cfg.reject_trials_by_epoch_max
        any_level_b = (
            cfg.reject_by_trial_mean_spread
            or cfg.reject_by_trial_max_spread
            or cfg.max_nan_trial_ratio is not None
        )

        if (any_level_a or any_level_b) and (epochs_a.shape[0] + epochs_b.shape[0]) > 0:
            n_a = epochs_a.shape[0]
            epochs_pooled = np.concatenate([epochs_a, epochs_b], axis=0)

            if any_level_a:
                nan_tc_mask = np.zeros((epochs_pooled.shape[0], len(feature_names)), dtype=bool)
                if cfg.reject_trials_by_epoch_mean:
                    nan_tc_mask |= detect_outlier_trial_channel_pairs_by_mean(
                        epochs_pooled,
                        threshold_factor=cfg.epoch_mean_threshold_factor,
                    )
                if cfg.reject_trials_by_epoch_max:
                    nan_tc_mask |= detect_outlier_trial_channel_pairs_by_max(
                        epochs_pooled,
                        threshold_factor=cfg.epoch_max_threshold_factor,
                    )
                if np.any(nan_tc_mask):
                    epochs_pooled = apply_trial_nan_mask(epochs_pooled, nan_tc_mask)
                    for ch_idx, ch_name in enumerate(feature_names):
                        trial_indices = [int(t) for t in np.where(nan_tc_mask[:, ch_idx])[0]]
                        if trial_indices:
                            excluded_trial_channel_pairs[ch_name] = trial_indices

            if any_level_b:
                reason_masks: dict[str, np.ndarray] = {}
                ch_excl_mask = np.zeros(len(feature_names), dtype=bool)
                if cfg.reject_by_trial_mean_spread:
                    mask = reject_channels_by_trial_mean_spread(
                        epochs_pooled,
                        threshold_factor=cfg.trial_mean_spread_threshold,
                    )
                    ch_excl_mask |= mask
                    reason_masks["trial_mean_spread"] = mask
                if cfg.reject_by_trial_max_spread:
                    mask = reject_channels_by_trial_max_spread(
                        epochs_pooled,
                        threshold_factor=cfg.trial_max_spread_threshold,
                    )
                    ch_excl_mask |= mask
                    reason_masks["trial_max_spread"] = mask
                if cfg.max_nan_trial_ratio is not None:
                    mask = reject_channels_by_nan_trial_ratio(
                        epochs_pooled,
                        max_ratio=cfg.max_nan_trial_ratio,
                    )
                    ch_excl_mask |= mask
                    reason_masks["nan_trial_ratio"] = mask
                if np.any(ch_excl_mask):
                    for ch_idx, ch_name in enumerate(feature_names):
                        if ch_excl_mask[ch_idx]:
                            reasons = [
                                reason for reason, mask in reason_masks.items() if mask[ch_idx]
                            ]
                            excluded_channels[ch_name] = ",".join(reasons)
                    for ch_name in excluded_channels:
                        excluded_trial_channel_pairs.pop(ch_name, None)
                    epochs_pooled, feature_names, _ = apply_channel_exclusions(
                        epochs_pooled,
                        list(feature_names),
                        ch_excl_mask,
                    )
                    if atlas_mode and feature_indices is not None:
                        feature_indices = [
                            indices
                            for indices, excl in zip(feature_indices, ch_excl_mask)
                            if not excl
                        ]

            epochs_a = epochs_pooled[:n_a]
            epochs_b = epochs_pooled[n_a:]

        state["excluded_channels"] = excluded_channels
        state["excluded_trial_channel_pairs"] = excluded_trial_channel_pairs
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
            "epoch_cleaning": json.loads(self.params.epoch_cleaning.model_dump_json()),
        }

    def _compute_and_build_result(
        self,
        context: TrialStatsProcessingContext,
    ) -> RegressionProcessingResult:
        predictor_a_raw_array = np.asarray(context.state["predictor_a_raw"], dtype=np.float64)
        predictor_b_raw_array = np.asarray(context.state["predictor_b_raw"], dtype=np.float64)
        predictor_a_transformed_array = np.asarray(
            context.state["predictor_a_transformed"], dtype=np.float64
        )
        predictor_b_transformed_array = np.asarray(
            context.state["predictor_b_transformed"], dtype=np.float64
        )
        kept_trials_a = list(context.kept_trials_a)
        kept_trials_b = list(context.kept_trials_b)

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

        from gin_bids_py_analysis.processing.utils.statistics import correct_p_values

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

        condition_a_epoch_means = self._compute_epoch_means(context.epochs_a, n_features)
        condition_b_epoch_means = self._compute_epoch_means(context.epochs_b, n_features)

        return RegressionProcessingResult(
            **self._build_common_result_kwargs(context),
            condition_a_slope=condition_a_slope,
            condition_a_intercept=condition_a_intercept,
            condition_a_r_value=condition_a_r_value,
            condition_a_p_value=condition_a_p_value,
            condition_a_p_value_corrected=condition_a_p_value_corrected,
            condition_a_significant_mask=condition_a_significant_mask,
            condition_b_slope=condition_b_slope,
            condition_b_intercept=condition_b_intercept,
            condition_b_r_value=condition_b_r_value,
            condition_b_p_value=condition_b_p_value,
            condition_b_p_value_corrected=condition_b_p_value_corrected,
            condition_b_significant_mask=condition_b_significant_mask,
            condition_a_mean=condition_a_mean,
            condition_b_mean=condition_b_mean,
            condition_a_sem=condition_a_sem,
            condition_b_sem=condition_b_sem,
            condition_a_trials_used=int(context.epochs_a.shape[0]),
            condition_b_trials_used=int(context.epochs_b.shape[0]),
            condition_a_predictor_raw_values=predictor_a_raw_array,
            condition_b_predictor_raw_values=predictor_b_raw_array,
            condition_a_predictor_transformed_values=predictor_a_transformed_array,
            condition_b_predictor_transformed_values=predictor_b_transformed_array,
            condition_a_predictor_values=predictor_a_effective_array,
            condition_b_predictor_values=predictor_b_effective_array,
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
            condition_a_stats_valid=condition_a_stats_valid,
            condition_b_stats_valid=condition_b_stats_valid,
            stats_valid=bool(condition_a_stats_valid or condition_b_stats_valid),
            condition_a_epoch_means=condition_a_epoch_means,
            condition_b_epoch_means=condition_b_epoch_means,
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

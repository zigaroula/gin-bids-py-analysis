from __future__ import annotations

from dataclasses import replace
import warnings

import numpy as np

from bidsforge.processing.utils.regression_stats import (
    compute_linear_regression_maps,
    compute_permuted_regression_maps,
    zscore_predictor_values_by_scope,
)
from bidsforge.processing.utils.statistics import correct_p_values
from bidsforge.processing.utils.trial_resolver import ResolvedTrial, TrialResolver

from ..processor import (
    BaseTimeFrequencyStatsProcessing,
    TimeFrequencyStatsContext,
    flatten_tf_epochs,
    unflatten_tf_map,
)
from ..result import TFConditionEstimate, TFConditionEpochs, TFConditionPair
from .params import TimeFrequencyRegressionParams
from .result import (
    TFConditionPredictorValues,
    TFConditionRegressionStats,
    TFPredictorValues,
    TFRegressionStats,
    TimeFrequencyRegressionResult,
)


class TimeFrequencyRegressionProcessing(BaseTimeFrequencyStatsProcessing):
    """Compute per-condition trial-wise regression maps from TFR derivatives."""

    def __init__(self, params: TimeFrequencyRegressionParams, resolver: TrialResolver) -> None:
        super().__init__(params=params, resolver=resolver)

    @property
    def params(self) -> TimeFrequencyRegressionParams:  # type: ignore[override]
        return self._params

    @params.setter
    def params(self, value: TimeFrequencyRegressionParams) -> None:
        self._params = value

    def _normalize_trials(self, trials: list[ResolvedTrial]) -> list[ResolvedTrial]:
        normalized = super()._normalize_trials(trials)
        out: list[ResolvedTrial] = []
        for trial in normalized:
            metadata = dict(trial.metadata)
            raw_value = _to_float_or_nan(metadata.get(self.params.predictor))
            transform = self.params.predictor_transform_by_condition.get(
                str(trial.label or ""),
            )
            scale = float(transform.scale) if transform is not None else 1.0
            offset = float(transform.offset) if transform is not None else 0.0
            transformed = raw_value * scale + offset if np.isfinite(raw_value) else np.nan
            metadata["predictor_raw_value"] = raw_value
            metadata["predictor_transform_scale"] = scale
            metadata["predictor_transform_offset"] = offset
            metadata["predictor_transformed_value"] = transformed
            metadata["predictor_value"] = transformed
            if trial.keep and not np.isfinite(transformed):
                out.append(
                    replace(
                        trial,
                        keep=False,
                        exclusion_reason=trial.exclusion_reason or "invalid_predictor_value",
                        metadata=metadata,
                    )
                )
            else:
                out.append(replace(trial, metadata=metadata))
        return out

    def _build_pipeline_metadata(self) -> dict[str, object]:
        return {
            "analysis_type": "time_frequency_regression",
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
        context: TimeFrequencyStatsContext,
    ) -> TimeFrequencyRegressionResult:
        flat_a = flatten_tf_epochs(context.epochs_a)
        flat_b = flatten_tf_epochs(context.epochs_b)
        n_channels = len(context.channel_names)
        n_freqs = int(len(context.frequency_hz))
        n_features = n_channels * n_freqs
        n_times = int(len(context.time_axis_s))

        raw_a = _trial_values(context.kept_trials_a, "predictor_raw_value")
        raw_b = _trial_values(context.kept_trials_b, "predictor_raw_value")
        transformed_a = _trial_values(context.kept_trials_a, "predictor_transformed_value")
        transformed_b = _trial_values(context.kept_trials_b, "predictor_transformed_value")
        values_a, values_b, z_valid_a, z_valid_b = zscore_predictor_values_by_scope(
            transformed_a,
            transformed_b,
            scope=self.params.predictor_zscore,
        )
        _store_predictor_values(context.kept_trials_a, values_a)
        _store_predictor_values(context.kept_trials_b, values_b)

        cond_a_ready = flat_a.shape[0] >= self.params.min_trials_per_condition and z_valid_a
        cond_b_ready = flat_b.shape[0] >= self.params.min_trials_per_condition and z_valid_b
        stats_a = _fit_condition(
            values_a if cond_a_ready else np.array([], dtype=np.float64),
            flat_a if cond_a_ready else flat_a[:0],
            n_features=n_features,
            n_times=n_times,
            n_channels=n_channels,
            n_freqs=n_freqs,
            alpha=self.params.significance_alpha,
            correction=self.params.p_value_correction_method,
        )
        stats_b = _fit_condition(
            values_b if cond_b_ready else np.array([], dtype=np.float64),
            flat_b if cond_b_ready else flat_b[:0],
            n_features=n_features,
            n_times=n_times,
            n_channels=n_channels,
            n_freqs=n_freqs,
            alpha=self.params.significance_alpha,
            correction=self.params.p_value_correction_method,
        )

        if self.params.n_permutations > 0:
            rng = np.random.default_rng(self.params.permutation_seed)
            if stats_a.stats_valid:
                stats_a.permuted_slopes = _permuted_slopes_tf(
                    values_a,
                    flat_a,
                    n_perm=self.params.n_permutations,
                    rng=rng,
                    n_features=n_features,
                    n_times=n_times,
                    n_channels=n_channels,
                    n_freqs=n_freqs,
                )
            if stats_b.stats_valid:
                stats_b.permuted_slopes = _permuted_slopes_tf(
                    values_b,
                    flat_b,
                    n_perm=self.params.n_permutations,
                    rng=rng,
                    n_features=n_features,
                    n_times=n_times,
                    n_channels=n_channels,
                    n_freqs=n_freqs,
                )

        mean_a, sem_a = _mean_sem_tf(context.epochs_a, n_channels, n_freqs, n_times)
        mean_b, sem_b = _mean_sem_tf(context.epochs_b, n_channels, n_freqs, n_times)

        kwargs = self._common_result_kwargs(context)
        kwargs["epochs"] = TFConditionEpochs(
            condition_a=context.epochs_a,
            condition_b=context.epochs_b,
        )
        return TimeFrequencyRegressionResult(
            **kwargs,
            signal_activity=TFConditionPair(
                condition_a=TFConditionEstimate(mean=mean_a, sem=sem_a),
                condition_b=TFConditionEstimate(mean=mean_b, sem=sem_b),
            ),
            stats_valid=bool(stats_a.stats_valid or stats_b.stats_valid),
            regression=TFRegressionStats(condition_a=stats_a, condition_b=stats_b),
            predictor_values=TFPredictorValues(
                condition_a=TFConditionPredictorValues(
                    raw_values=raw_a,
                    transformed_values=transformed_a,
                    values=values_a,
                ),
                condition_b=TFConditionPredictorValues(
                    raw_values=raw_b,
                    transformed_values=transformed_b,
                    values=values_b,
                ),
            ),
            predictor=self.params.predictor,
            predictor_zscore=self.params.predictor_zscore,
            predictor_transform_by_condition={
                condition: {
                    "scale": float(transform.scale),
                    "offset": float(transform.offset),
                }
                for condition, transform in self.params.predictor_transform_by_condition.items()
            },
        )


def _fit_condition(
    predictor_values: np.ndarray,
    epochs: np.ndarray,
    *,
    n_features: int,
    n_times: int,
    n_channels: int,
    n_freqs: int,
    alpha: float,
    correction: str,
) -> TFConditionRegressionStats:
    slope, intercept, r_value, p_value, valid, n_obs = compute_linear_regression_maps(
        predictor_values,
        epochs,
        n_features=n_features,
        n_times=n_times,
        return_n_obs=True,
        nan_policy="pointwise",
    )
    p_corr = correct_p_values(p_value, method=correction)
    sig = np.isfinite(p_corr) & (p_corr < alpha)
    t_values = _r_to_t_values_by_n_obs(r_value, n_obs)
    return TFConditionRegressionStats(
        slope=unflatten_tf_map(slope, n_channels=n_channels, n_freqs=n_freqs),
        intercept=unflatten_tf_map(intercept, n_channels=n_channels, n_freqs=n_freqs),
        r_value=unflatten_tf_map(r_value, n_channels=n_channels, n_freqs=n_freqs),
        t_values=unflatten_tf_map(t_values, n_channels=n_channels, n_freqs=n_freqs),
        p_value=unflatten_tf_map(p_value, n_channels=n_channels, n_freqs=n_freqs),
        p_value_corrected=unflatten_tf_map(
            p_corr,
            n_channels=n_channels,
            n_freqs=n_freqs,
        ),
        significant_mask=unflatten_tf_map(
            sig.astype(np.float64),
            n_channels=n_channels,
            n_freqs=n_freqs,
        ).astype(bool),
        n_trials_used=int(epochs.shape[0]),
        stats_valid=bool(valid),
    )


def _permuted_slopes_tf(
    predictor_values: np.ndarray,
    epochs: np.ndarray,
    *,
    n_perm: int,
    rng: np.random.Generator,
    n_features: int,
    n_times: int,
    n_channels: int,
    n_freqs: int,
) -> np.ndarray:
    flat = compute_permuted_regression_maps(
        predictor_values,
        epochs,
        n_perm=n_perm,
        rng=rng,
        n_features=n_features,
        n_times=n_times,
    )
    return flat.reshape(n_perm, n_channels, n_freqs, n_times)


def _r_to_t_values_by_n_obs(r_value: np.ndarray, n_obs: np.ndarray) -> np.ndarray:
    out = np.full_like(r_value, np.nan, dtype=np.float64)
    df = np.asarray(n_obs, dtype=np.float64) - 2.0
    valid = np.isfinite(r_value) & (np.abs(r_value) < 1.0) & np.isfinite(df) & (df > 0.0)
    out[valid] = r_value[valid] * np.sqrt(df[valid] / (1.0 - np.square(r_value[valid])))
    perfect = np.isfinite(r_value) & (np.abs(r_value) >= 1.0)
    out[perfect] = np.sign(r_value[perfect]) * np.inf
    return out


def _mean_sem_tf(
    epochs: np.ndarray,
    n_channels: int,
    n_freqs: int,
    n_times: int,
) -> tuple[np.ndarray, np.ndarray]:
    if epochs.shape[0] == 0:
        empty = np.full((n_channels, n_freqs, n_times), np.nan, dtype=np.float64)
        return empty.copy(), empty.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(epochs, axis=0, dtype=np.float64)
        sem = (
            np.nanstd(epochs, axis=0, ddof=1, dtype=np.float64) / np.sqrt(float(epochs.shape[0]))
            if epochs.shape[0] >= 2
            else np.full((n_channels, n_freqs, n_times), np.nan, dtype=np.float64)
        )
    return mean, sem


def _trial_values(trials: list[ResolvedTrial], key: str) -> np.ndarray:
    return np.asarray([_to_float_or_nan(trial.metadata.get(key)) for trial in trials], dtype=np.float64)


def _store_predictor_values(trials: list[ResolvedTrial], values: np.ndarray) -> None:
    for trial, value in zip(trials, np.asarray(values, dtype=np.float64).ravel()):
        trial.metadata["predictor_value"] = float(value) if np.isfinite(value) else np.nan


def _to_float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import ttest_1samp, ttest_ind

from bidsforge.processing.utils.statistics import correct_p_values
from bidsforge.processing.utils.trial_resolver import TrialResolver

from ..processor import (
    BaseTimeFrequencyStatsProcessing,
    TimeFrequencyStatsContext,
    flatten_tf_epochs,
    unflatten_tf_map,
)
from ..result import TFConditionEstimate, TFConditionEpochs, TFConditionPair
from .params import TimeFrequencyConditionTestParams
from .result import (
    TFConditionContrast,
    TFDifferenceEstimate,
    TFGrandAverage,
    TimeFrequencyConditionTestResult,
)


class TimeFrequencyConditionTestProcessing(BaseTimeFrequencyStatsProcessing):
    """Compute subject-level condition A vs B maps from TFR derivatives."""

    def __init__(
        self,
        params: TimeFrequencyConditionTestParams,
        resolver: TrialResolver,
    ) -> None:
        super().__init__(params=params, resolver=resolver)

    @property
    def params(self) -> TimeFrequencyConditionTestParams:  # type: ignore[override]
        return self._params

    @params.setter
    def params(self, value: TimeFrequencyConditionTestParams) -> None:
        self._params = value

    def _build_pipeline_metadata(self) -> dict[str, object]:
        return {
            "analysis_type": "time_frequency_condition_test",
            "equal_var": bool(self.params.equal_var),
            "compute_grand_average": bool(self.params.compute_grand_average),
        }

    def _compute_and_build_result(
        self,
        context: TimeFrequencyStatsContext,
    ) -> TimeFrequencyConditionTestResult:
        flat_a = flatten_tf_epochs(context.epochs_a)
        flat_b = flatten_tf_epochs(context.epochs_b)
        n_channels = len(context.channel_names)
        n_freqs = int(len(context.frequency_hz))
        n_features = n_channels * n_freqs
        n_times = int(len(context.time_axis_s))
        stats_valid = (
            flat_a.shape[0] >= self.params.min_trials_per_condition
            and flat_b.shape[0] >= self.params.min_trials_per_condition
        )

        mean_a, sem_a = _mean_sem_tf(context.epochs_a, n_channels, n_freqs, n_times)
        mean_b, sem_b = _mean_sem_tf(context.epochs_b, n_channels, n_freqs, n_times)
        difference_mean = mean_a - mean_b
        difference_sem = np.sqrt(np.square(sem_a) + np.square(sem_b))

        empty = np.full((n_features, n_times), np.nan, dtype=np.float64)
        if stats_valid:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                stats = ttest_ind(
                    flat_a,
                    flat_b,
                    axis=0,
                    equal_var=self.params.equal_var,
                    nan_policy="omit",
                )
            t_flat = np.asarray(stats.statistic, dtype=np.float64)
            p_raw_flat = np.asarray(stats.pvalue, dtype=np.float64)
        else:
            t_flat = empty.copy()
            p_raw_flat = empty.copy()

        p_flat = correct_p_values(
            p_raw_flat,
            method=self.params.p_value_correction_method,
        )
        sig_flat = np.isfinite(p_flat) & (p_flat < self.params.significance_alpha)

        permuted = None
        if self.params.n_permutations > 0 and stats_valid:
            permuted = _compute_permuted_t_values(
                flat_a,
                flat_b,
                n_perm=self.params.n_permutations,
                n_channels=n_channels,
                n_freqs=n_freqs,
                rng=np.random.default_rng(self.params.permutation_seed),
                equal_var=self.params.equal_var,
            )

        grand_average = None
        if self.params.compute_grand_average:
            grand_average = _compute_grand_average(
                context.power,
                n_channels=n_channels,
                n_freqs=n_freqs,
                n_times=n_times,
            )

        kwargs = self._common_result_kwargs(context)
        kwargs["epochs"] = TFConditionEpochs(
            condition_a=context.epochs_a,
            condition_b=context.epochs_b,
        )
        return TimeFrequencyConditionTestResult(
            **kwargs,
            signal_activity=TFConditionPair(
                condition_a=TFConditionEstimate(mean=mean_a, sem=sem_a),
                condition_b=TFConditionEstimate(mean=mean_b, sem=sem_b),
            ),
            stats_valid=stats_valid,
            difference=TFDifferenceEstimate(mean=difference_mean, sem=difference_sem),
            contrast=TFConditionContrast(
                t_values=unflatten_tf_map(t_flat, n_channels=n_channels, n_freqs=n_freqs),
                p_values=unflatten_tf_map(p_flat, n_channels=n_channels, n_freqs=n_freqs),
                p_values_uncorrected=unflatten_tf_map(
                    p_raw_flat,
                    n_channels=n_channels,
                    n_freqs=n_freqs,
                ),
                significant_mask=unflatten_tf_map(
                    sig_flat.astype(np.float64),
                    n_channels=n_channels,
                    n_freqs=n_freqs,
                ).astype(bool),
                permuted_t_values=permuted,
            ),
            grand_average=grand_average,
        )


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
        if epochs.shape[0] < 2:
            sem = np.full_like(mean, np.nan, dtype=np.float64)
        else:
            sem = np.nanstd(epochs, axis=0, ddof=1, dtype=np.float64) / np.sqrt(
                float(epochs.shape[0])
            )
    return mean, sem


def _compute_permuted_t_values(
    flat_a: np.ndarray,
    flat_b: np.ndarray,
    *,
    n_perm: int,
    n_channels: int,
    n_freqs: int,
    rng: np.random.Generator,
    equal_var: bool,
) -> np.ndarray:
    pooled = np.concatenate([flat_a, flat_b], axis=0)
    n_a = flat_a.shape[0]
    n_total = pooled.shape[0]
    out = np.empty((n_perm, n_channels, n_freqs, flat_a.shape[2]), dtype=np.float32)
    for idx in range(n_perm):
        order = rng.permutation(n_total)
        perm_a = pooled[order[:n_a]]
        perm_b = pooled[order[n_a:]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            stats = ttest_ind(
                perm_a,
                perm_b,
                axis=0,
                equal_var=equal_var,
                nan_policy="omit",
            )
        out[idx] = unflatten_tf_map(
            np.asarray(stats.statistic, dtype=np.float64),
            n_channels=n_channels,
            n_freqs=n_freqs,
        ).astype(np.float32)
    return out


def _compute_grand_average(
    power: np.ndarray,
    *,
    n_channels: int,
    n_freqs: int,
    n_times: int,
) -> TFGrandAverage:
    if power.shape[0] == 0:
        empty = np.full((n_channels, n_freqs, n_times), np.nan, dtype=np.float64)
        return TFGrandAverage(mean=empty.copy(), std=empty.copy(), t_values=empty.copy(), p_values=empty.copy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mean = np.nanmean(power, axis=0, dtype=np.float64)
        std = np.nanstd(power, axis=0, ddof=1, dtype=np.float64)
        stats = ttest_1samp(power, 0.0, axis=0, nan_policy="omit")
    return TFGrandAverage(
        mean=mean,
        std=std,
        t_values=np.asarray(stats.statistic, dtype=np.float64),
        p_values=np.asarray(stats.pvalue, dtype=np.float64),
    )


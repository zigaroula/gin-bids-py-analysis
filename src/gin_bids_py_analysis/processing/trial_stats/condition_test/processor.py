"""Processor for per-subject condition-A vs condition-B statistics on iEEG derivatives."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from gin_bids_py_analysis.processing.utils.statistics import (
    compute_condition_sem,
    correct_p_values,
)
from gin_bids_py_analysis.processing.utils.trial_annotator import TrialWindowAnnotator
from gin_bids_py_analysis.processing.utils.trial_resolver import ResolvedTrial, TrialResolver

from ..processor import BaseTrialStatsProcessing, TrialStatsProcessingContext
from ..result import ActivityEstimate, ConditionActivity
from .params import ConditionTestParams
from .result import ConditionContrast, ConditionTestProcessingResult, DifferenceEstimate
from .stats import (
    compute_bootstrap_difference_ci95,
    compute_condition_statistics,
    compute_duration_channel_significance,
    compute_permuted_statistics,
    compute_single_bin_channel_significance,
)


class ConditionTestProcessing(BaseTrialStatsProcessing):
    """Compute per-subject condition_a-vs-condition_b statistics on iEEG data."""

    def __init__(
        self,
        params: ConditionTestParams,
        resolver: TrialResolver,
        annotators: Sequence[TrialWindowAnnotator] = (),
    ) -> None:
        super().__init__(params=params, resolver=resolver, annotators=annotators)

    @property
    def params(self) -> ConditionTestParams:  # type: ignore[override]
        return self._params

    @params.setter
    def params(self, value: ConditionTestParams) -> None:
        self._params = value

    def _normalize_trials(
        self,
        *,
        group,
        ieeg_file,
        raw,
        anchor_events,
        trials: list[ResolvedTrial],
    ) -> list[ResolvedTrial]:
        del group, ieeg_file, raw, anchor_events
        normalized: list[ResolvedTrial] = []
        supported_labels = {self.params.condition_a, self.params.condition_b}
        for trial in trials:
            if not trial.keep:
                normalized.append(trial)
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
                        metadata=dict(trial.metadata),
                    )
                )
            else:
                normalized.append(trial)
        return normalized

    def _build_pipeline_metadata(self, *, state: dict[str, object]) -> dict[str, object]:
        del state
        return {
            "equal_var": self.params.equal_var,
            "n_permutations": self.params.n_permutations,
            "channel_significance_mode": self.params.channel_significance_mode,
            "channel_significance_duration_threshold_ms": self.params.channel_significance_duration_threshold_ms,
        }

    def _compute_and_build_result(
        self,
        context: TrialStatsProcessingContext,
    ) -> ConditionTestProcessingResult:
        stats_valid = (
            context.epochs_a.shape[0] >= self.params.min_trials_per_condition
            and context.epochs_b.shape[0] >= self.params.min_trials_per_condition
        )
        n_features = len(context.feature_names)
        n_times = len(context.time_axis_eval)
        t_values, p_values_raw, mean_a, mean_b, mean_difference = (
            compute_condition_statistics(
                context.epochs_a if stats_valid else context.epochs_a[:0],
                context.epochs_b if stats_valid else context.epochs_b[:0],
                n_channels=n_features,
                n_times=n_times,
                equal_var=self.params.equal_var,
            )
        )
        if context.epochs_a.size:
            mean_a = np.nanmean(context.epochs_a, axis=0, dtype=np.float64)
        if context.epochs_b.size:
            mean_b = np.nanmean(context.epochs_b, axis=0, dtype=np.float64)
        mean_difference = mean_a - mean_b
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
        difference_sem = (condition_a_sem ** 2 + condition_b_sem ** 2) ** 0.5
        difference_ci95_low, difference_ci95_high = compute_bootstrap_difference_ci95(
            context.epochs_a,
            context.epochs_b,
            n_bootstraps=2000,
            random_state=self.params.permutation_seed,
        )
        p_values = correct_p_values(
            p_values_raw,
            method=self.params.p_value_correction_method,
        )

        permuted_t_values = None
        if self.params.n_permutations > 0 and stats_valid:
            rng = np.random.default_rng(self.params.permutation_seed)
            permuted_t_values = compute_permuted_statistics(
                context.epochs_a,
                context.epochs_b,
                self.params.n_permutations,
                rng,
                equal_var=self.params.equal_var,
            )

        significant_mask = (p_values < self.params.significance_alpha) & np.isfinite(p_values)

        channel_significant_mask = None
        if self.params.channel_significance_mode == "single_bin" and stats_valid:
            channel_significant_mask = compute_single_bin_channel_significance(
                context.epochs_a,
                context.epochs_b,
                equal_var=self.params.equal_var,
                p_value_correction_method=self.params.p_value_correction_method,
                significance_alpha=self.params.significance_alpha,
            )
        elif self.params.channel_significance_mode == "duration" and stats_valid:
            channel_significant_mask = compute_duration_channel_significance(
                significant_mask,
                context.time_axis_eval,
                threshold_ms=self.params.channel_significance_duration_threshold_ms,
            )

        return ConditionTestProcessingResult(
            **self._build_common_result_kwargs(context),
            activity=ConditionActivity(
                condition_a=ActivityEstimate(mean=mean_a, sem=condition_a_sem),
                condition_b=ActivityEstimate(mean=mean_b, sem=condition_b_sem),
            ),
            stats_valid=stats_valid,
            difference=DifferenceEstimate(
                mean=mean_difference,
                sem=difference_sem,
                ci95_low=difference_ci95_low,
                ci95_high=difference_ci95_high,
            ),
            contrast=ConditionContrast(
                t_values=t_values,
                p_values=p_values,
                p_values_uncorrected=p_values_raw,
                significant_mask=significant_mask,
                permuted_t_values=permuted_t_values,
                channel_significant_mask=channel_significant_mask,
            ),
        )

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from bidsforge.processing.base import BaseProcessingResult
from bidsforge.processing.utils.serialization import OutputTree
from bidsforge.processing.utils.trial_resolver import ResolvedTrial


# ---------------------------------------------------------------------------
# Shared signal-activity dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SignalActivityEstimate:
    """Per-condition mean and SEM for a single condition."""

    mean: np.ndarray = field(default_factory=lambda: np.array([]))
    sem: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ConditionSignalActivity:
    """Paired mean/SEM signal activity for both conditions."""

    condition_a: SignalActivityEstimate = field(default_factory=SignalActivityEstimate)
    condition_b: SignalActivityEstimate = field(default_factory=SignalActivityEstimate)


@dataclass
class ConditionEpochs:
    """Raw epoch arrays for both conditions (optional; 3-D when present)."""

    condition_a: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class ConditionTrialSummaryValues:
    """Per-trial activity-summary values for both conditions (channels × trials)."""

    condition_a: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b: np.ndarray = field(default_factory=lambda: np.array([]))


# ---------------------------------------------------------------------------
# Module-level helpers (also used by writer.py for TSV serialization)
# ---------------------------------------------------------------------------


def _deep_merge(dest: dict[str, object], src: dict[str, object]) -> dict[str, object]:
    """Recursively merge *src* into *dest* in-place; *src* wins on conflicts."""
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dest.get(key), dict):
            _deep_merge(dest[key], value)  # type: ignore[arg-type]
        else:
            dest[key] = value
    return dest


def _condition_inputs_json(trial: object) -> str:
    metadata = getattr(trial, "metadata", {})
    value = metadata.get("condition_inputs", {}) if isinstance(metadata, dict) else {}
    return json.dumps(value, sort_keys=True, ensure_ascii=True, default=str)


def _to_float_or_nan(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out if np.isfinite(out) else float("nan")


# ---------------------------------------------------------------------------
# Base result
# ---------------------------------------------------------------------------


@dataclass
class BaseTrialStatsProcessingResult(BaseProcessingResult):
    """Shared subject-level result fields for trial statistics pipelines."""

    time_axis_s: np.ndarray = field(default_factory=lambda: np.array([]))
    channel_names: list[str] = field(default_factory=list)
    condition_a: str = "condition_a"
    condition_b: str = "condition_b"
    condition_a_trial_count: int = 0
    condition_b_trial_count: int = 0
    signal_activity: ConditionSignalActivity = field(default_factory=ConditionSignalActivity)
    sfreq: float = 0.0
    resolved_trials: list[ResolvedTrial] = field(default_factory=list)
    source_ieeg_files: list[str] = field(default_factory=list)
    source_table_files: list[str] = field(default_factory=list)
    source_electrodes_files: list[str] = field(default_factory=list)
    analysis_level: str = "channel"
    atlas_name: str | None = None
    atlas_regions: list[str] = field(default_factory=list)
    region_channels: dict[str, list[str]] = field(default_factory=dict)
    window_ms: float = 0.0
    n_bins: int = 0
    activity_zscore: str = "none"
    activity_baseline_tmin_s: float = -0.2
    activity_baseline_tmax_s: float = 0.0
    activity_baseline_scope: str = "global"
    activity_baseline_remove_outlier_trial_means: bool = False
    activity_baseline_outlier_method: str = "median_mad"
    p_value_correction_method: str = "fdr_bh"
    significance_alpha: float = 0.05
    trial_activity_summary_kind: str = "epoch_mean"
    trial_activity_summary_missing_response_policy: str = "clamp_to_epoch"
    trial_activity_summary_source: dict[str, str] = field(default_factory=dict)
    trial_activity_summary_label: str = "Epoch mean activity"
    epoch_cleaning_audit: dict[str, object] = field(default_factory=dict)
    stats_valid: bool = False
    epochs: ConditionEpochs = field(default_factory=ConditionEpochs)
    trial_activity_summary_values: ConditionTrialSummaryValues = field(
        default_factory=ConditionTrialSummaryValues
    )

    # ------------------------------------------------------------------
    # Output tree
    # ------------------------------------------------------------------

    def to_output_tree(
        self,
        *,
        include_epochs: bool = False,
        pipeline_name: str = "unknown",
        pipeline_version: str = "unknown",
    ) -> OutputTree:
        """Return the canonical serialisable output tree for this result.

        Subclasses should call ``super().to_output_tree(...)`` and deep-merge
        their analysis-specific subtrees into the returned dict.
        """
        primary_axis_name = "region" if self.analysis_level == "roi" else "channel"
        region_order = self.channel_names if self.analysis_level == "roi" else []
        ordered_regions = list(region_order)
        for region in self.region_channels:
            if region not in ordered_regions:
                ordered_regions.append(region)

        region_pairs: list[str] = []
        channel_pairs: list[str] = []
        for region in ordered_regions:
            for channel in self.region_channels.get(region, []):
                region_pairs.append(str(region))
                channel_pairs.append(str(channel))

        tree: dict[str, object] = {
            "data": {
                "signal_activity": {
                    self.condition_a: {
                        "mean": np.asarray(self.signal_activity.condition_a.mean, dtype=np.float64),
                        "sem": np.asarray(self.signal_activity.condition_a.sem, dtype=np.float64),
                    },
                    self.condition_b: {
                        "mean": np.asarray(self.signal_activity.condition_b.mean, dtype=np.float64),
                        "sem": np.asarray(self.signal_activity.condition_b.sem, dtype=np.float64),
                    },
                },
            },
            "axes": {
                primary_axis_name: np.array(self.channel_names, dtype=object),
                "time_s": np.asarray(self.time_axis_s, dtype=np.float64),
            },
            "meta": self._build_meta_tree(
                region_order=region_order,
                region_pairs=region_pairs,
                channel_pairs=channel_pairs,
            ),
            "trials": self._build_trial_table_tree(),
            "provenance": {
                "source_ieeg_files": np.array(self.source_ieeg_files, dtype=object),
                "source_table_files": np.array(self.source_table_files, dtype=object),
                "source_electrodes_files": np.array(
                    self.source_electrodes_files, dtype=object
                ),
                "pipeline_name": pipeline_name,
                "pipeline_version": pipeline_version,
            },
        }

        if include_epochs and self.epochs.condition_a.ndim == 3:
            tree["epochs"] = {
                self.condition_a: np.asarray(self.epochs.condition_a, dtype=np.float64),
                self.condition_b: np.asarray(self.epochs.condition_b, dtype=np.float64),
                "channel": np.array(self.channel_names, dtype=object),
                "time_s": np.asarray(self.time_axis_s, dtype=np.float64),
            }

        summary_a = np.asarray(self.trial_activity_summary_values.condition_a, dtype=np.float64)
        summary_b = np.asarray(self.trial_activity_summary_values.condition_b, dtype=np.float64)
        if summary_a.ndim == 2 and summary_b.ndim == 2:
            tree["trial_activity_summary"] = {
                f"{self.condition_a}_values": summary_a,
                f"{self.condition_b}_values": summary_b,
                "kind": str(self.trial_activity_summary_kind),
                "missing_response_policy": str(
                    self.trial_activity_summary_missing_response_policy
                ),
                "source_json": json.dumps(
                    self.trial_activity_summary_source,
                    sort_keys=True,
                    ensure_ascii=True,
                ),
                "label": str(self.trial_activity_summary_label),
            }

        return tree

    def _build_meta_tree(
        self,
        *,
        region_order: list[str],
        region_pairs: list[str],
        channel_pairs: list[str],
    ) -> dict[str, object]:
        return {
            "schema_name": "trial_stats_subject",
            "schema_version": "1.0",
            "condition_labels": np.array(
                [self.condition_a, self.condition_b], dtype=object
            ),
            "trial_counts": np.array(
                [self.condition_a_trial_count, self.condition_b_trial_count],
                dtype=np.int64,
            ),
            "sampling_frequency_hz": float(self.sfreq),
            "p_value_correction_method": str(self.p_value_correction_method),
            "significance_alpha": float(self.significance_alpha),
            "analysis_level": str(self.analysis_level),
            "atlas_name": str(self.atlas_name or ""),
            "atlas_regions": np.array(self.atlas_regions, dtype=object),
            "atlas_region_channel_map": {
                "region_order": np.array(region_order, dtype=object),
                "region": np.array(region_pairs, dtype=object),
                "channel": np.array(channel_pairs, dtype=object),
            },
            "window_ms": float(self.window_ms),
            "n_bins": int(self.n_bins),
            "activity_zscore": str(self.activity_zscore),
            "activity_baseline_tmin_s": float(self.activity_baseline_tmin_s),
            "activity_baseline_tmax_s": float(self.activity_baseline_tmax_s),
            "activity_baseline_scope": str(self.activity_baseline_scope),
            "activity_baseline_remove_outlier_trial_means": bool(
                self.activity_baseline_remove_outlier_trial_means
            ),
            "trial_activity_summary_kind": str(self.trial_activity_summary_kind),
            "trial_activity_summary_missing_response_policy": str(
                self.trial_activity_summary_missing_response_policy
            ),
            "trial_activity_summary_source_json": json.dumps(
                self.trial_activity_summary_source,
                sort_keys=True,
                ensure_ascii=True,
            ),
            "trial_activity_summary_label": str(self.trial_activity_summary_label),
            "epoch_cleaning_json": json.dumps(
                self.metadata.get("epoch_cleaning", {}),
                sort_keys=True,
                ensure_ascii=True,
            ),
            "epoch_cleaning_audit_json": json.dumps(
                self.epoch_cleaning_audit,
                sort_keys=True,
                ensure_ascii=True,
            ),
            "window_samples": int(self.metadata.get("window_samples", 0)),
            "effective_n_bins": int(
                self.metadata.get("effective_n_bins", len(self.time_axis_s))
            ),
            "binning_mode": str(
                self.metadata.get(
                    "binning_mode",
                    "window_ms"
                    if self.window_ms > 0
                    else ("n_bins" if self.n_bins > 0 else "none"),
                )
            ),
            "stats_valid": bool(self.stats_valid),
        }

    def _build_trial_table_tree(self) -> dict[str, object]:
        return {
            "source_file": np.array(
                [str(trial.source_file.path) for trial in self.resolved_trials],
                dtype=object,
            ),
            "anchor_event_index": np.array(
                [trial.anchor_event_index for trial in self.resolved_trials],
                dtype=np.int64,
            ),
            "anchor_event_code": np.array(
                [trial.anchor_event_code or "" for trial in self.resolved_trials],
                dtype=object,
            ),
            "anchor_onset_s": np.array(
                [trial.anchor_onset_s for trial in self.resolved_trials],
                dtype=np.float64,
            ),
            "resolved_label": np.array(
                [trial.label or "" for trial in self.resolved_trials],
                dtype=object,
            ),
            "keep": np.array(
                [trial.keep for trial in self.resolved_trials], dtype=bool
            ),
            "exclusion_reason": np.array(
                [trial.exclusion_reason or "" for trial in self.resolved_trials],
                dtype=object,
            ),
            "trial_id": np.array(
                [trial.trial_id or "" for trial in self.resolved_trials],
                dtype=object,
            ),
            "condition_inputs": np.array(
                [_condition_inputs_json(trial) for trial in self.resolved_trials],
                dtype=object,
            ),
            "condition_resolution_reason": np.array(
                [
                    str(trial.metadata.get("condition_resolution_reason", ""))
                    for trial in self.resolved_trials
                ],
                dtype=object,
            ),
            "trial_activity_summary_response_time_s": np.array(
                [
                    _to_float_or_nan(
                        trial.metadata.get("trial_activity_summary_response_time_s")
                    )
                    for trial in self.resolved_trials
                ],
                dtype=np.float64,
            ),
            "nan_masked_features": np.array(
                [
                    json.dumps(sorted(trial.metadata.get("nan_masked_features", [])))
                    for trial in self.resolved_trials
                ],
                dtype=object,
            ),
            "baseline_outlier_masked_features": np.array(
                [
                    json.dumps(
                        sorted(trial.metadata.get("baseline_outlier_masked_features", []))
                    )
                    for trial in self.resolved_trials
                ],
                dtype=object,
            ),
            "activity_summary_skip_reason": np.array(
                [
                    str(trial.metadata.get("activity_summary_skip_reason", ""))
                    for trial in self.resolved_trials
                ],
                dtype=object,
            ),
            **self._build_trial_table_extra_tree(),
        }

    def _build_trial_table_extra_tree(self) -> dict[str, object]:
        """Extension point for subclasses to add analysis-specific trial columns."""
        return {}

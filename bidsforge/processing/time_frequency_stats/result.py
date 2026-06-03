from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bidsforge.processing.base import BaseProcessingResult
from bidsforge.processing.utils.serialization import OutputTree, compressed
from bidsforge.processing.utils.trial_resolver import ResolvedTrial


@dataclass
class TFConditionEstimate:
    mean: np.ndarray = field(default_factory=lambda: np.array([]))
    sem: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class TFConditionPair:
    condition_a: TFConditionEstimate = field(default_factory=TFConditionEstimate)
    condition_b: TFConditionEstimate = field(default_factory=TFConditionEstimate)


@dataclass
class TFConditionEpochs:
    condition_a: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_b: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class BaseTimeFrequencyStatsResult(BaseProcessingResult):
    """Shared result fields for subject-level TF statistics."""

    channel_names: list[str] = field(default_factory=list)
    frequency_hz: np.ndarray = field(default_factory=lambda: np.array([]))
    time_axis_s: np.ndarray = field(default_factory=lambda: np.array([]))
    condition_a: str = "condition_a"
    condition_b: str = "condition_b"
    condition_a_trial_count: int = 0
    condition_b_trial_count: int = 0
    resolved_trials: list[ResolvedTrial] = field(default_factory=list)
    source_tfr_file: str = ""
    source_table_files: list[str] = field(default_factory=list)
    source_ieeg_files: list[str] = field(default_factory=list)
    source_electrodes_files: list[str] = field(default_factory=list)
    signal_activity: TFConditionPair = field(default_factory=TFConditionPair)
    epochs: TFConditionEpochs = field(default_factory=TFConditionEpochs)
    stats_valid: bool = False
    power_mode: str = "stored"
    p_value_correction_method: str = "fdr_bh"
    significance_alpha: float = 0.05

    def _base_output_tree(
        self,
        *,
        include_epochs: bool,
        pipeline_name: str,
        pipeline_version: str,
    ) -> OutputTree:
        tree: dict[str, Any] = {
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
                "channel": np.asarray(self.channel_names, dtype=object),
                "frequency_hz": np.asarray(self.frequency_hz, dtype=np.float64),
                "time_s": np.asarray(self.time_axis_s, dtype=np.float64),
            },
            "meta": {
                "schema_name": "time_frequency_stats_subject",
                "schema_version": "1.0",
                "analysis_type": str(self.metadata.get("analysis_type", "")),
                "condition_labels": np.asarray([self.condition_a, self.condition_b], dtype=object),
                "trial_counts": np.asarray(
                    [self.condition_a_trial_count, self.condition_b_trial_count],
                    dtype=np.int64,
                ),
                "stats_valid": bool(self.stats_valid),
                "power_mode": str(self.power_mode),
                "p_value_correction_method": str(self.p_value_correction_method),
                "significance_alpha": float(self.significance_alpha),
                "time_window_s": np.asarray(
                    self.metadata.get("time_window_s", []), dtype=np.float64
                ),
                "time_selection": str(self.metadata.get("time_selection", "")),
                "baseline_grand_average": bool(
                    self.metadata.get("baseline_grand_average", False)
                ),
                "n_permutations": int(self.metadata.get("n_permutations", 0)),
            },
            "trials": self._trial_table_tree(),
            "provenance": {
                "source_tfr_file": str(self.source_tfr_file),
                "source_table_files": np.asarray(self.source_table_files, dtype=object),
                "source_ieeg_files": np.asarray(self.source_ieeg_files, dtype=object),
                "source_electrodes_files": np.asarray(
                    self.source_electrodes_files,
                    dtype=object,
                ),
                "pipeline_name": pipeline_name,
                "pipeline_version": pipeline_version,
            },
        }
        if include_epochs and self.epochs.condition_a.ndim == 4:
            tree["epochs"] = {
                self.condition_a: compressed(
                    np.asarray(self.epochs.condition_a, dtype=np.float32)
                ),
                self.condition_b: compressed(
                    np.asarray(self.epochs.condition_b, dtype=np.float32)
                ),
            }
        return tree

    def _trial_table_tree(self) -> dict[str, object]:
        return {
            "source_file": np.asarray(
                [str(trial.source_file.path) for trial in self.resolved_trials],
                dtype=object,
            ),
            "anchor_event_index": np.asarray(
                [trial.anchor_event_index for trial in self.resolved_trials],
                dtype=np.int64,
            ),
            "anchor_event_code": np.asarray(
                [trial.anchor_event_code or "" for trial in self.resolved_trials],
                dtype=object,
            ),
            "anchor_onset_s": np.asarray(
                [trial.anchor_onset_s for trial in self.resolved_trials],
                dtype=np.float64,
            ),
            "trial_id": np.asarray(
                [trial.trial_id or "" for trial in self.resolved_trials],
                dtype=object,
            ),
            "resolved_label": np.asarray(
                [trial.label or "" for trial in self.resolved_trials],
                dtype=object,
            ),
            "keep": np.asarray([trial.keep for trial in self.resolved_trials], dtype=bool),
            "exclusion_reason": np.asarray(
                [trial.exclusion_reason or "" for trial in self.resolved_trials],
                dtype=object,
            ),
            "condition_inputs": np.asarray(
                [
                    json.dumps(
                        trial.metadata.get("condition_inputs", {}),
                        sort_keys=True,
                        ensure_ascii=True,
                        default=str,
                    )
                    for trial in self.resolved_trials
                ],
                dtype=object,
            ),
            **self._trial_table_extra_tree(),
        }

    def _trial_table_extra_tree(self) -> dict[str, object]:
        return {}

from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from gin_bids_py_analysis.processing.base import BaseProcessingResult, BaseProcessingWriter
from gin_bids_py_analysis.processing.utils.serialization import (
    OutputTree,
    write_hdf5_tree,
    write_matlab_tree,
)

from .result import (
    BaseTrialStatsProcessingResult,
    _condition_inputs_json,
    _to_float_or_nan,
)


def _package_version() -> str:
    try:
        return version("gin-bids-py-analysis")
    except PackageNotFoundError:
        return "unknown"


class BaseTrialStatsProcessingWriter(BaseProcessingWriter, ABC):
    """Shared writer helpers for subject-level trial statistics outputs."""

    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        if not isinstance(result, BaseTrialStatsProcessingResult):
            raise TypeError(
                f"Expected BaseTrialStatsProcessingResult, got {type(result).__name__!r}"
            )

        self._validate_result(result)
        tree = result.to_output_tree(
            include_epochs=getattr(self.params, "include_epochs", False),
            pipeline_name=self._pipeline_name(),
            pipeline_version=_package_version(),
        )
        if self.params.output_format == "matlab":
            write_matlab_tree(output_path, tree, root_name=self._pipeline_name())
        else:
            write_hdf5_tree(output_path, tree)
        self._write_trial_table_tsv(result, _trial_table_path(output_path))

    def _write_trial_table_tsv(
        self,
        result: BaseTrialStatsProcessingResult,
        output_path: Path,
    ) -> None:
        with open(output_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(
                self._trial_table_prefix_header()
                + self._trial_table_shared_extra_header()
                + self._trial_table_extra_header()
                + self._trial_table_suffix_header()
            )
            for trial in result.resolved_trials:
                writer.writerow(
                    self._trial_table_prefix_row(trial)
                    + self._trial_table_shared_extra_row(trial)
                    + self._trial_table_extra_row(trial)
                    + self._trial_table_suffix_row(trial)
                )

    def _trial_table_prefix_header(self) -> list[str]:
        return [
            "source_file",
            "anchor_event_index",
            "anchor_event_code",
            "anchor_onset_s",
            "trial_id",
            "resolved_label",
        ]

    def _trial_table_prefix_row(self, trial: object) -> list[object]:
        return [
            str(getattr(getattr(trial, "source_file", None), "path", "")),
            getattr(trial, "anchor_event_index", ""),
            getattr(trial, "anchor_event_code", "") or "",
            getattr(trial, "anchor_onset_s", ""),
            getattr(trial, "trial_id", "") or "",
            getattr(trial, "label", "") or "",
        ]

    def _trial_table_extra_header(self) -> list[str]:
        return []

    def _trial_table_extra_row(self, trial: object) -> list[object]:
        del trial
        return []

    def _trial_table_shared_extra_header(self) -> list[str]:
        return [
            "trial_activity_summary_response_time_s",
            "nan_masked_features",
            "baseline_outlier_masked_features",
            "activity_summary_skip_reason",
        ]

    def _trial_table_shared_extra_row(self, trial: object) -> list[object]:
        metadata = getattr(trial, "metadata", {})
        if not isinstance(metadata, dict):
            return [float("nan"), "[]", "[]", ""]
        rt = _to_float_or_nan(metadata.get("trial_activity_summary_response_time_s"))
        nan_features = metadata.get("nan_masked_features", [])
        nan_features_str = json.dumps(sorted(nan_features)) if nan_features else "[]"
        baseline_features = metadata.get("baseline_outlier_masked_features", [])
        baseline_features_str = (
            json.dumps(sorted(baseline_features)) if baseline_features else "[]"
        )
        skip_reason = str(metadata.get("activity_summary_skip_reason", ""))
        return [rt, nan_features_str, baseline_features_str, skip_reason]

    def _trial_table_suffix_header(self) -> list[str]:
        return [
            "condition_inputs",
            "condition_resolution_reason",
            "keep",
            "exclusion_reason",
        ]

    def _trial_table_suffix_row(self, trial: object) -> list[object]:
        metadata = getattr(trial, "metadata", {})
        return [
            _condition_inputs_json(trial),
            (
                str(metadata.get("condition_resolution_reason", ""))
                if isinstance(metadata, dict)
                else ""
            ),
            str(getattr(trial, "keep", False)).lower(),
            getattr(trial, "exclusion_reason", "") or "",
        ]

    @abstractmethod
    def _pipeline_name(self) -> str:
        """Return the subject-level provenance pipeline name."""

    def _validate_result(self, result: BaseTrialStatsProcessingResult) -> None:
        del result


def _trial_table_path(output_path: Path) -> Path:
    tsv_path = output_path.with_suffix(".tsv")
    return tsv_path.with_name(tsv_path.name.replace("_stats.tsv", "_trials.tsv"))


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

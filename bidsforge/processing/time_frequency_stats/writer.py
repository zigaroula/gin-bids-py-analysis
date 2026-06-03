from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np

from bidsforge.processing.base import BaseProcessingResult, BaseProcessingWriter
from bidsforge.processing.utils.serialization import write_hdf5_tree, write_matlab_tree

from .result import BaseTimeFrequencyStatsResult


def _package_version() -> str:
    try:
        return version("bidsforge")
    except PackageNotFoundError:
        return "unknown"


class BaseTimeFrequencyStatsWriter(BaseProcessingWriter, ABC):
    """Shared writer for subject-level TF statistics."""

    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        if not isinstance(result, BaseTimeFrequencyStatsResult):
            raise TypeError(
                f"Expected BaseTimeFrequencyStatsResult, got {type(result).__name__!r}"
            )
        tree = result.to_output_tree(
            include_epochs=bool(getattr(self.params, "include_epochs", False)),
            pipeline_name=self._pipeline_name(),
            pipeline_version=_package_version(),
        )
        if self.params.output_format == "matlab":
            write_matlab_tree(output_path, tree, root_name=self._pipeline_name())
        else:
            write_hdf5_tree(output_path, tree)
        self._write_trial_table(result, _trial_table_path(output_path))

    def _write_trial_table(self, result: BaseTimeFrequencyStatsResult, path: Path) -> None:
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter="\t")
            writer.writerow(
                [
                    "source_file",
                    "anchor_event_index",
                    "anchor_event_code",
                    "anchor_onset_s",
                    "trial_id",
                    "resolved_label",
                    "keep",
                    "exclusion_reason",
                    "condition_inputs",
                ]
                + self._trial_table_extra_header()
            )
            for trial in result.resolved_trials:
                writer.writerow(
                    [
                        str(trial.source_file.path),
                        trial.anchor_event_index,
                        trial.anchor_event_code or "",
                        trial.anchor_onset_s,
                        trial.trial_id or "",
                        trial.label or "",
                        str(trial.keep).lower(),
                        trial.exclusion_reason or "",
                        json.dumps(
                            trial.metadata.get("condition_inputs", {}),
                            sort_keys=True,
                            ensure_ascii=True,
                            default=str,
                        ),
                    ]
                    + self._trial_table_extra_row(trial)
                )

    def _trial_table_extra_header(self) -> list[str]:
        return []

    def _trial_table_extra_row(self, trial: object) -> list[object]:
        del trial
        return []

    @abstractmethod
    def _pipeline_name(self) -> str:
        """Return the pipeline provenance/root name."""


def _trial_table_path(output_path: Path) -> Path:
    tsv_path = output_path.with_suffix(".tsv")
    return tsv_path.with_name(tsv_path.name.replace("_stats.tsv", "_trials.tsv"))


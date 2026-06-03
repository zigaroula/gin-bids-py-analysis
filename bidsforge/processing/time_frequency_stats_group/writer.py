from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from bidsforge.processing.base import BaseProcessingResult, BaseProcessingWriter
from bidsforge.processing.utils.serialization import write_hdf5_tree, write_matlab_tree

from .result import BaseTimeFrequencyStatsGroupResult


def package_version() -> str:
    try:
        return version("bidsforge")
    except PackageNotFoundError:
        return "unknown"


class BaseTimeFrequencyStatsGroupWriter(BaseProcessingWriter):
    def _build_output_path(self, entities: dict) -> Path:
        group_entities = dict(entities)
        group_entities["subject"] = "group"
        return super()._build_output_path(group_entities)

    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        if not isinstance(result, BaseTimeFrequencyStatsGroupResult):
            raise TypeError(
                f"Expected BaseTimeFrequencyStatsGroupResult, got {type(result).__name__!r}"
            )
        tree = result.to_output_tree(
            pipeline_name=self.params.pipeline_label,
            pipeline_version=package_version(),
        )
        if self.params.output_format == "matlab":
            write_matlab_tree(output_path, tree, root_name=self.params.pipeline_label)
        else:
            write_hdf5_tree(output_path, tree)

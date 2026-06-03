from __future__ import annotations

from pathlib import Path

from bidsforge.processing.base import BaseProcessingResult
from bidsforge.processing.utils.serialization import write_hdf5_tree, write_matlab_tree

from ..writer import BaseTimeFrequencyStatsGroupWriter, package_version
from .result import TimeFrequencyRegressionGroupResult


class TimeFrequencyRegressionGroupWriter(BaseTimeFrequencyStatsGroupWriter):
    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        if not isinstance(result, TimeFrequencyRegressionGroupResult):
            raise TypeError(
                f"Expected TimeFrequencyRegressionGroupResult, got {type(result).__name__!r}"
            )
        tree = result.to_output_tree(
            pipeline_name="time_frequency_regression_group",
            pipeline_version=package_version(),
        )
        if self.params.output_format == "matlab":
            write_matlab_tree(output_path, tree, root_name="time_frequency_regression_group")
        else:
            write_hdf5_tree(output_path, tree)

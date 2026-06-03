from __future__ import annotations

from pathlib import Path

from bidsforge.processing.base import BaseProcessingResult

from ..writer import BaseTimeFrequencyStatsGroupWriter, package_version
from .result import TimeFrequencyConditionTestGroupResult


class TimeFrequencyConditionTestGroupWriter(BaseTimeFrequencyStatsGroupWriter):
    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        if not isinstance(result, TimeFrequencyConditionTestGroupResult):
            raise TypeError(
                f"Expected TimeFrequencyConditionTestGroupResult, got {type(result).__name__!r}"
            )
        tree = result.to_output_tree(
            pipeline_name="time_frequency_condition_test_group",
            pipeline_version=package_version(),
        )
        if self.params.output_format == "matlab":
            from bidsforge.processing.utils.serialization import write_matlab_tree

            write_matlab_tree(output_path, tree, root_name="time_frequency_condition_test_group")
        else:
            from bidsforge.processing.utils.serialization import write_hdf5_tree

            write_hdf5_tree(output_path, tree)

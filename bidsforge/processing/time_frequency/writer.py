from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from bidsforge.processing.base import BaseProcessingResult, BaseProcessingWriter
from bidsforge.processing.utils.serialization import write_hdf5_tree, write_matlab_tree

from .result import TimeFrequencyProcessingResult


def _package_version() -> str:
    try:
        return version("bidsforge")
    except PackageNotFoundError:
        return "unknown"


class TimeFrequencyProcessingWriter(BaseProcessingWriter):
    def _write_data(self, result: BaseProcessingResult, output_path) -> None:
        if not isinstance(result, TimeFrequencyProcessingResult):
            raise TypeError(
                f"Expected TimeFrequencyProcessingResult, got {type(result).__name__!r}"
            )
        tree = result.to_output_tree(pipeline_version=_package_version())
        if self.params.output_format == "matlab":
            write_matlab_tree(output_path, tree, root_name="time_frequency")
        else:
            write_hdf5_tree(output_path, tree)

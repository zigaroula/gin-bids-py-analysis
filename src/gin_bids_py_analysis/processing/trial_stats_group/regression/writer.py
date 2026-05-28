from __future__ import annotations

from pathlib import Path

from gin_bids_py_analysis.processing.base import BaseProcessingResult
from gin_bids_py_analysis.processing.utils.serialization import (
    write_hdf5_tree,
    write_matlab_tree,
)

from ..writer import BaseTrialStatsGroupProcessingWriter, package_version
from .result import RegressionGroupProcessingResult


class RegressionGroupProcessingWriter(BaseTrialStatsGroupProcessingWriter):
    """Write group-level regression ROI outputs to HDF5 or MATLAB."""

    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        if not isinstance(result, RegressionGroupProcessingResult):
            raise TypeError(
                f"Expected RegressionGroupProcessingResult, got {type(result).__name__!r}"
            )

        tree = result.to_output_tree(
            pipeline_name="regression_group",
            pipeline_version=package_version(),
        )
        if self.params.output_format == "matlab":
            write_matlab_tree(output_path, tree, root_name="regression_group")
        else:
            write_hdf5_tree(output_path, tree)

from __future__ import annotations

from pathlib import Path

from gin_bids_py_analysis.processing.base import BaseProcessingResult, BaseProcessingWriter

from .result import HilbertProcessingResult


class HilbertProcessingWriter(BaseProcessingWriter):
    """
    Writes :class:`HilbertProcessingResult` to a BIDS derivatives folder.

    Path construction, directory creation, and BIDS naming are all handled
    by :meth:`BaseProcessingWriter.write`.  This class only needs to declare
    the three pipeline constants and implement :meth:`_write_data`.

    Example::

        writer = HilbertProcessingWriter(Path("/data/my_study"))
        out_path = writer.write(result)
    """

    PIPELINE_LABEL = "hilbert"
    OUTPUT_SUFFIX = "hilbert"
    OUTPUT_EXTENSION = ".npy"  # TODO: update when output format is finalised

    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        if not isinstance(result, HilbertProcessingResult):
            raise TypeError(
                f"Expected HilbertProcessingResult, got {type(result).__name__!r}"
            )
        # TODO: implement serialisation once HilbertProcessingResult.output is typed.
        # e.g.: np.save(output_path, result.output)
        raise NotImplementedError(
            "HilbertProcessingWriter._write_data() is a stub — implement once "
            "HilbertProcessingResult.output type is finalised."
        )

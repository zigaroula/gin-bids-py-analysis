from __future__ import annotations

from pathlib import Path

from gin_bids_py_analysis.processing.base import BaseProcessingResult, BaseProcessingWriter

from .result import HilbertProcessingResult


class HilbertProcessingWriter(BaseProcessingWriter):
    """
    Writes :class:`HilbertProcessingResult` to a BIDS derivatives folder.

    Path construction, directory creation, and BIDS naming are all handled
    by :meth:`BaseProcessingWriter.write`.  This class only needs to
    implement :meth:`_write_data`.

    Pass a :class:`~gin_bids_py_analysis.processing.hilbert.HilbertWriterParams`
    instance to the constructor — only ``bids_root`` is required, the pipeline
    routing fields default to Hilbert-appropriate values.

    Example::

        from pathlib import Path
        from gin_bids_py_analysis.processing.hilbert import (
            HilbertWriterParams, HilbertProcessingWriter,
        )

        writer = HilbertProcessingWriter(
            HilbertWriterParams(bids_root=Path("/data/my_study"))
        )
        out_path = writer.write(result)
    """

    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        if not isinstance(result, HilbertProcessingResult):
            raise TypeError(
                f"Expected HilbertProcessingResult, got {type(result).__name__!r}"
            )
        
        # write placeholder empty file: implement once the output format and serialisation strategy are finalised
        output_path.touch()

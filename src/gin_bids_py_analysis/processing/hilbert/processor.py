from __future__ import annotations

from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.base import BaseProcessing

from .params import HilbertParams
from .result import HilbertProcessingResult


class HilbertProcessing(BaseProcessing):
    """
    Hilbert-transform-based analysis processor.

    Instantiate with a :class:`HilbertParams` object defining the frequency
    bands and sampling frequency; then call :meth:`run` or :meth:`execute`
    with a list of :class:`~gin_bids_py_analysis.bids.file_group.BIDSFileGroup`.

    Example::

        params = HilbertParams(freq_bands=[(1, 4), (8, 12)], sfreq=1000.0)
        processor = HilbertProcessing(params)
        groups = [BIDSFileGroup(primary=f) for f in ieeg_files]
        out_paths = processor.run(groups, writer, output_root)
    """

    def __init__(self, params: HilbertParams) -> None:
        self.params = params

    def process_group(self, group: BIDSFileGroup) -> HilbertProcessingResult:
        """
        Run the Hilbert analysis on one file group.

        For single-file analyses, ``group.primary`` is the iEEG file.
        For multi-modal analyses, ``group.secondaries`` may carry companion
        files (e.g. physio recordings).

        TODO: Implement once the data loader interface (``gin_bids_py_analysis.data``)
              is available and the algorithm is defined.
        """
        raise NotImplementedError(
            "HilbertProcessing.process_group() is a stub — implement once the "
            "data loader interface is available in gin_bids_py_analysis.data."
        )

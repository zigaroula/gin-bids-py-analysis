"""
Delphos HFO/spike detection processing subpackage.

Public API:
    - DelphosParams: Algorithm parameters for the Delphos detector
    - DelphosWriterParams: Writer configuration for output files
    - DelphosProcessing: Main processor orchestrating detection
    - DelphosProcessingResult: Container for detection results
    - DelphosProcessingWriter: BIDS derivatives writer for detection output
    - MontageMode, BipolarDirection, BipolarStorage: Montage configuration
"""

__version__ = "0.1.0"

from gin_bids_py_analysis.processing.utils.channels import (
    BipolarDirection,
    BipolarStorage,
    MontageMode,
)

from .params import DelphosParams, DelphosWriterParams
from .processor import DelphosProcessing
from .result import DelphosProcessingResult
from .writer import DelphosProcessingWriter

__all__ = [
    "BipolarDirection",
    "BipolarStorage",
    "DelphosParams",
    "DelphosProcessing",
    "DelphosProcessingResult",
    "DelphosProcessingWriter",
    "DelphosWriterParams",
    "MontageMode",
]

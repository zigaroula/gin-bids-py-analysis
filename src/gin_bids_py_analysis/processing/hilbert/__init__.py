"""Hilbert processing module for BIDS data. Based on Localizer's implementation: https://github.com/CRNL-Eduwell/Localizer"""

__version__ = "0.1.0"

from .params import (
    BipolarDirection,
    BipolarStorage,
    HilbertParams,
    HilbertWriterParams,
    MontageMode,
    NormalizationMode,
    ProcessingMethod,
)
from .processor import HilbertProcessing
from .result import HilbertProcessingResult
from .writer import HilbertProcessingWriter

__all__ = [
    "BipolarDirection",
    "BipolarStorage",
    "HilbertParams",
    "HilbertProcessing",
    "HilbertProcessingResult",
    "HilbertProcessingWriter",
    "HilbertWriterParams",
    "MontageMode",
    "NormalizationMode",
    "ProcessingMethod",
]

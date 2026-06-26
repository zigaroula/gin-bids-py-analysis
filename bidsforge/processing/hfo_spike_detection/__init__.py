"""
HFO/spike detection processing subpackage.

Public API:
    - HfoSpikeDetectorParams: Algorithm parameters for the HFO/spike detector
    - HfoSpikeDetectorWriterParams: Writer configuration for output files
    - HfoSpikeDetectorProcessing: Main processor orchestrating detection
    - HfoSpikeDetectorProcessingResult: Container for detection results
    - HfoSpikeDetectorProcessingWriter: BIDS derivatives writer for detection output
    - MontageMode, BipolarDirection, BipolarStorage: Montage configuration
"""

__version__ = "1.1.0"

from bidsforge.processing.utils.channels import (
    BipolarDirection,
    BipolarStorage,
    MontageMode,
)

from .params import HfoSpikeDetectorParams, HfoSpikeDetectorWriterParams
from .processor import HfoSpikeDetectorProcessing
from .result import HfoSpikeDetectorProcessingResult
from .result_loader import load_hfo_spike_detection_result
from .writer import HfoSpikeDetectorProcessingWriter

__all__ = [
    "BipolarDirection",
    "BipolarStorage",
    "HfoSpikeDetectorParams",
    "HfoSpikeDetectorProcessing",
    "HfoSpikeDetectorProcessingResult",
    "HfoSpikeDetectorProcessingWriter",
    "HfoSpikeDetectorWriterParams",
    "load_hfo_spike_detection_result",
    "MontageMode",
]

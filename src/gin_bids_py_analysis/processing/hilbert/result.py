from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gin_bids_py_analysis.bids.file_group import BIDSFileGroup  # noqa: F401 — re-exported for subclassers
from gin_bids_py_analysis.processing.base import BaseProcessingResult


@dataclass
class HilbertProcessingResult(BaseProcessingResult):
    """
    Result of a Hilbert-transform analysis.

    TODO: Replace the ``output`` placeholder with typed numpy arrays for
          amplitude envelopes and/or instantaneous phase once the algorithm
          and data loader interfaces are defined.
    """

    # Placeholder — replace with e.g. ``envelope: np.ndarray``
    output: Any = field(default=None)

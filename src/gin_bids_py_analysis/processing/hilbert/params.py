from pydantic import Field

from gin_bids_py_analysis.processing.base import BaseProcessingParams, BaseWriterParams


class HilbertParams(BaseProcessingParams):
    """
    Parameters for Hilbert-transform-based analysis.

    TODO: Expand with the full set of algorithm parameters once the iEEG
          data loader interface is defined (e.g. epoch length, overlap,
          baseline correction strategy).
    """

    freq_bands: list[tuple[float, float]] = Field(
        default_factory=list,
        description="List of (low_hz, high_hz) frequency band tuples.",
    )
    sfreq: float = Field(
        default=1000.0,
        gt=0,
        description="Sampling frequency in Hz.",
    )

class HilbertWriterParams(BaseWriterParams):
    """
    Writer parameters for the Hilbert analysis pipeline.

    Provides defaults for the three output-routing fields so that only
    ``bids_root`` needs to be supplied at construction time.

    Example::

        from pathlib import Path
        from gin_bids_py_analysis.processing.hilbert import HilbertWriterParams, HilbertProcessingWriter

        writer = HilbertProcessingWriter(HilbertWriterParams(bids_root=Path("/data/my_study")))

    TODO: add ``output_format: Literal["npy", "mat"] = "npy"`` once the
    serialisation strategy is finalised.
    """

    pipeline_label: str = "hilbert"
    output_suffix: str = "hilbert"
    output_extension: str = ".h5"


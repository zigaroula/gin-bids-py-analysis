from pydantic import BaseModel, Field


class HilbertParams(BaseModel):
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

from __future__ import annotations

from pydantic import Field, model_validator

from gin_bids_py_analysis.processing.base import BaseProcessingParams, BaseWriterParams
from gin_bids_py_analysis.processing.utils.channels import (
    BipolarDirection,
    BipolarStorage,
    MontageMode,
)


class HilbertParams(BaseProcessingParams):
    """
    Parameters for the Hilbert-band envelope pipeline.

    Frequency bands are defined implicitly by a uniform grid of bin edges:
    ``[f_min, f_min+f_step, f_min+2·f_step, …, f_max]``.  Adjacent bin
    pairs form the subbands that are band-pass filtered, e.g. bins
    ``[50, 60, 70]`` produce two subbands: 50-60 Hz and 60-70 Hz.

    If the highest bin exceeds the Nyquist frequency (``fs/2``) the grid is
    silently clamped (Shannon clamp) before processing.

    Example::

        params = HilbertParams(f_min=50, f_max=150, f_step=10)
        # → bins [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        # → subbands 50-60, 60-70, …, 140-150
    """

    # ------------------------------------------------------------------
    # Frequency grid
    # ------------------------------------------------------------------

    f_min: float = Field(gt=0, description="Lowest bin edge in Hz.")
    f_max: float = Field(gt=0, description="Highest bin edge in Hz.")
    f_step: float = Field(gt=0, description="Step between adjacent bin edges in Hz.")

    # ------------------------------------------------------------------
    # Downsampling
    # ------------------------------------------------------------------

    downsampled_frequency_hz: float = Field(
        default=64.0,
        gt=0,
        description=(
            "Target sampling rate for the envelope output in Hz.  "
            "``scipy.signal.resample_poly`` is used to achieve the exact target rate "
            "with anti-aliasing."
        ),
    )

    # ------------------------------------------------------------------
    # Montage
    # ------------------------------------------------------------------

    montage_mode: MontageMode = Field(
        default=MontageMode.MONO,
        description="Channel referencing mode applied before processing.",
    )
    bipolar_direction: BipolarDirection = Field(
        default=BipolarDirection.NEXT_MINUS_PREVIOUS,
        description=(
            "Which contact is subtracted from which.  "
            "Only used when ``montage_mode`` is ``BIPOLAR``."
        ),
    )
    bipolar_storage: BipolarStorage = Field(
        default=BipolarStorage.PREVIOUS,
        description=(
            "Naming convention for the derived bipolar channel.  "
            "Only used when ``montage_mode`` is ``BIPOLAR``."
        ),
    )
    channels_for_montage: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of channel names to include in the montage.  "
            "If ``None``, all channels are used."
        ),
    )

    # ------------------------------------------------------------------
    # Post-processing toggles
    # ------------------------------------------------------------------

    smoothing_windows_ms: list[int] = Field(
        default=[0, 250, 500, 1000, 2500, 5000],
        description=(
            "Moving-average window durations in milliseconds.  "
            "``0`` means no smoothing (identity); all other values apply the "
            "causal moving average."
        ),
    )
    do_downsample: bool = Field(
        default=True,
        description="Decimate envelopes to ``downsampled_frequency_hz`` before normalization.",
    )
    do_normalize_percent: bool = Field(
        default=True,
        description="Express each subband envelope as a percentage of its middle-50%% baseline.",
    )
    do_smoothing: bool = Field(
        default=True,
        description="Apply moving-average smoothing to the averaged envelope.",
    )
    centered: bool = Field(
        default=False,
        description=(
            "If ``True``, subtract 100 from all outputs so the baseline is 0 "
            "rather than 100.  Has no effect when ``do_normalize_percent`` is ``False``."
        ),
    )

    @model_validator(mode="after")
    def _check_frequency_range(self) -> "HilbertParams":
        if self.f_max <= self.f_min:
            raise ValueError("f_max must be greater than f_min.")
        if self.f_step > (self.f_max - self.f_min):
            raise ValueError("f_step must be smaller than (f_max - f_min).")
        return self


class HilbertWriterParams(BaseWriterParams):
    """
    Writer parameters for the Hilbert analysis pipeline.

    Only ``bids_root`` must be supplied; all routing fields default to
    Hilbert-appropriate values.  Output files are written as HDF5 (``.h5``),
    with one dataset per smoothing window.

    Example::

        from pathlib import Path
        from gin_bids_py_analysis.processing.hilbert import HilbertWriterParams, HilbertProcessingWriter

        writer = HilbertProcessingWriter(HilbertWriterParams(bids_root=Path("/data/my_study")))
    """

    pipeline_label: str = "hilbert"
    output_modality: str = "ieeg"
    output_suffix: str = "ieeg"
    output_description: str = "hilbert"
    output_extension: str = ".h5"


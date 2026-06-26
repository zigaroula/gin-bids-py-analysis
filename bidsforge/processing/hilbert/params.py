from __future__ import annotations

from enum import Enum
import math
from typing import Literal

from pydantic import Field, field_validator, model_validator

from bidsforge.processing.base import BaseProcessingParams, BaseWriterParams
from bidsforge.processing.utils.channels import (
    BipolarDirection,
    BipolarStorage,
    MontageMode,
)
from bidsforge.processing.utils.input_events import EventSource


class NormalizationMode(str, Enum):
    """Normalization applied to the Hilbert-band envelope after downsampling.

    * ``NONE`` - no normalization; output unit is signal amplitude.
    * ``PERCENT`` - express as percentage of mid-recording baseline
      (baseline = mean of middle 50% of the signal; baseline region -> 100).
    * ``PERCENT_CENTERED`` - same as ``PERCENT``, then subtract 100 so the
      baseline region is centered at 0 instead of 100.
    * ``DB`` - express in decibels relative to mid-recording baseline:
      ``20 * log10(amplitude / baseline)``; baseline region -> 0 dB.
    """

    NONE = "none"
    PERCENT = "percent"
    PERCENT_CENTERED = "percent_centered"
    DB = "db"

    @property
    def unit_label(self) -> str:
        """Human-readable unit string stored in output file metadata."""
        return {
            NormalizationMode.NONE: "amplitude",
            NormalizationMode.PERCENT: "percent",
            NormalizationMode.PERCENT_CENTERED: "percent",
            NormalizationMode.DB: "dB",
        }[self]

    @property
    def unit(self) -> str:
        """Unit string written into BrainVision ``.vhdr`` channel headers."""
        return {
            NormalizationMode.NONE: "µV",
            NormalizationMode.PERCENT: "%",
            NormalizationMode.PERCENT_CENTERED: "%",
            NormalizationMode.DB: "dB",
        }[self]

    @property
    def scale_factor(self) -> float:
        """Multiplicative scale applied to data before BrainVision export.

        BrainVision conventions expect µV; for amplitude data the raw MNE
        values are in V so we multiply by 1e-6 to convert. Normalized
        outputs (percent, dB) are dimensionless and require no scaling.
        """
        return 1e-6 if self == NormalizationMode.NONE else 1.0

    @property
    def is_percent(self) -> bool:
        """``True`` for both ``PERCENT`` and ``PERCENT_CENTERED`` modes."""
        return self in (NormalizationMode.PERCENT, NormalizationMode.PERCENT_CENTERED)

    @property
    def is_centered(self) -> bool:
        """``True`` only for ``PERCENT_CENTERED``; used to apply the -100 offset."""
        return self == NormalizationMode.PERCENT_CENTERED


class ProcessingMethod(str, Enum):
    """Order of operations used in the Hilbert-band envelope pipeline.

    * ``LOCALIZER`` - normalise and smooth *after* downsampling (default).
    * ``SPM2ENV`` - normalise and smooth *at the native recording frequency*,
      then downsample.
    """

    LOCALIZER = "localizer"
    SPM2ENV = "spm2env"


class HilbertParams(BaseProcessingParams):
    """Parameters for the Hilbert-band envelope pipeline.

    Frequency bands are defined implicitly by a uniform grid of bin edges:
    ``[f_min, f_min+f_step, f_min+2*f_step, ..., f_max]``. Adjacent bin
    pairs form the subbands that are band-pass filtered, e.g. bins
    ``[50, 60, 70]`` produce two subbands: 50-60 Hz and 60-70 Hz.

    If the highest bin exceeds the Nyquist frequency (``fs/2``), the grid is
    silently clamped (Shannon clamp) before processing.

    The ``method`` field selects the order of operations: ``LOCALIZER``
    (default) downsamples first and then normalises and smooths; ``SPM2ENV``
    normalises and smooths at the native recording frequency and downsamples
    last.

    Example::

        params = HilbertParams(f_min=50, f_max=150, f_step=10)
        # -> bins [50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
        # -> subbands 50-60, 60-70, ..., 140-150
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

    computation_frequency_hz: float | None = Field(
        default=None,
        gt=0,
        description=(
            "Sampling rate at which BPF and Hilbert computations are performed in Hz. "
            "When set, the signal is downsampled to this rate *before* filtering. "
            "Set to ``None`` (default) to run computations at the native recording "
            "sampling rate."
        ),
    )
    downsampled_frequency_hz: float | None = Field(
        default=64.0,
        gt=0,
        description=(
            "Target sampling rate for the envelope output in Hz. "
            "``scipy.signal.resample_poly`` is used to achieve the exact target rate "
            "with anti-aliasing. Set to ``None`` to skip downsampling and keep the "
            "envelopes at the original recording sampling rate."
        ),
    )
    event_sample_shift_samples: int = Field(
        default=0,
        description=(
            "Temporary compatibility offset applied to annotation/event onsets "
            "in source-sampling-rate samples before exporting events at the "
            "envelope sampling rate. This does not shift the signal itself."
        ),
    )
    events_source: EventSource = Field(
        default="annotations",
        description=(
            "Where continuous-file events are read from. 'annotations' reads the "
            "signal file annotations; 'events_tsv' requires a matching _events.tsv "
            "attached in BIDSFileGroup.secondaries; 'auto' prefers that secondary "
            "_events.tsv and falls back to annotations."
        ),
    )
    notch_filter_freqs: list[float] = Field(
        default_factory=list,
        description=(
            "Optional notch-filter frequencies in Hz applied to the continuous Raw "
            "before Hilbert band-pass/envelope extraction. Empty disables notch filtering."
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
            "Which contact is subtracted from which. "
            "Only used when ``montage_mode`` is ``BIPOLAR``."
        ),
    )
    bipolar_storage: BipolarStorage = Field(
        default=BipolarStorage.PREVIOUS,
        description=(
            "Naming convention for the derived bipolar channel. "
            "Only used when ``montage_mode`` is ``BIPOLAR``."
        ),
    )
    channels_for_montage: list[str] | str | None = Field(
        default=None,
        description=(
            "Channel selector applied before montage and processing. "
            "Pass a list of exact channel names to keep, or a regex string "
            "matched via ``re.fullmatch`` against each channel name "
            "(e.g. ``r'[A-Za-z]p?([1-9]|1[0-9])'`` for one-letter prefix, "
            "optional 'p', index 1-19). If ``None``, all channels are used."
        ),
    )
    channels_to_exclude_for_montage: list[str] | str | None = Field(
        default=None,
        description=(
            "Channel selector applied after ``channels_for_montage`` and "
            "before montage/processing. Pass a list of exact channel names "
            "to remove, or a regex string matched via ``re.fullmatch`` "
            "against each channel name. If ``None``, no channels are excluded."
        ),
    )

    # ------------------------------------------------------------------
    # Post-processing toggles
    # ------------------------------------------------------------------

    smoothing_windows_ms: list[int] = Field(
        default=[0],
        description=(
            "Sliding-average window durations in milliseconds. ``0`` means no "
            "smoothing (identity); all other values apply ``moving_average``. "
            "Use ``[0]`` to disable smoothing entirely."
        ),
    )
    normalization_mode: NormalizationMode = Field(
        default=NormalizationMode.PERCENT,
        description=(
            "Normalization applied to the envelope after downsampling. "
            "``PERCENT`` expresses each subband as a percentage of its middle-50%% "
            "baseline; ``PERCENT_CENTERED`` does the same and then subtracts 100 so "
            "the baseline region sits at 0; ``DB`` uses "
            "``20*log10(amplitude / baseline)``; ``NONE`` skips normalization."
        ),
    )
    method: ProcessingMethod = Field(
        default=ProcessingMethod.LOCALIZER,
        description=(
            "Order of operations in the envelope pipeline. "
            "``LOCALIZER`` (default) downsamples first, then normalises and smooths. "
            "``SPM2ENV`` normalises and smooths at the native recording frequency, "
            "then downsamples."
        ),
    )

    @field_validator("notch_filter_freqs", mode="before")
    @classmethod
    def _coerce_notch_filter_freqs(cls, value: object) -> list[float]:
        if value in (None, ""):
            return []
        values: list[object]
        if isinstance(value, (str, int, float)):
            if isinstance(value, str) and "," in value:
                values = [item.strip() for item in value.split(",")]
            else:
                values = [value]
        else:
            try:
                values = list(value)  # type: ignore[arg-type]
            except TypeError as exc:
                raise ValueError("notch_filter_freqs must be a number or a list of numbers.") from exc

        freqs: list[float] = []
        for item in values:
            if item in (None, ""):
                continue
            try:
                freq = float(item)
            except (TypeError, ValueError) as exc:
                raise ValueError("notch_filter_freqs must contain only numeric frequencies.") from exc
            if not math.isfinite(freq) or freq <= 0.0:
                raise ValueError("notch_filter_freqs values must be finite and strictly positive.")
            freqs.append(freq)
        return freqs

    @model_validator(mode="after")
    def _check_frequency_range(self) -> "HilbertParams":
        if self.f_max <= self.f_min:
            raise ValueError("f_max must be greater than f_min.")
        if self.f_step > (self.f_max - self.f_min):
            raise ValueError("f_step must be smaller than (f_max - f_min).")
        return self


class HilbertWriterParams(BaseWriterParams):
    """Writer parameters for the Hilbert analysis pipeline.

    Only ``bids_root`` must be supplied; all routing fields default to
    Hilbert-appropriate values. Use ``output_format`` to choose the output
    backend:

    * ``"hdf5"`` (default) - writes a single ``.h5`` file per recording.
    * ``"brainvision"`` - writes one ``.vhdr`` / ``.vmrk`` / ``.eeg`` triplet
      per smoothing window.

    Example::

        from pathlib import Path
        from bidsforge.processing.hilbert import HilbertWriterParams, HilbertProcessingWriter

        writer = HilbertProcessingWriter(HilbertWriterParams(bids_root=Path("/data/my_study")))
    """

    pipeline_label: str = "hilbert"
    output_modality: str = "ieeg"
    output_suffix: str = "ieeg"
    output_description: str = "hilbert"
    output_format: Literal["hdf5", "brainvision", "matlab"] = Field(
        default="hdf5",
        description=(
            "Output backend: 'hdf5' (default) writes a single .h5 file; "
            "'brainvision' writes a .vhdr/.vmrk/.eeg triplet per smoothing window; "
            "'matlab' writes a .mat file readable by MATLAB/scipy."
        ),
    )

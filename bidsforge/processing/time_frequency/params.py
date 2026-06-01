from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, model_validator

from bidsforge.processing.base import BaseProcessingParams, BaseWriterParams
from bidsforge.processing.utils.channels import (
    BipolarDirection,
    BipolarStorage,
    MontageMode,
)
from bidsforge.processing.utils.input_events import EventSource


class TimeFrequencyMethod(str, Enum):
    """Spectral estimation backend used by the time-frequency pipeline."""

    FIELDTRIP = "fieldtrip"
    """Replicate FieldTrip ``ft_specest_mtmconvol`` conventions."""


class TimeFrequencyParams(BaseProcessingParams):
    """Parameters for the subject-level time-frequency power pipeline."""

    anchor_event_codes: list[str] = Field(
        default_factory=list,
        description="Event codes used as epoch anchors.",
    )
    tmin_s: float = Field(
        default=-1.5,
        description="Epoch start in seconds relative to the anchor event.",
    )
    tmax_s: float = Field(
        default=2.0,
        description="Epoch end in seconds relative to the anchor event.",
    )
    drop_partial_epochs: bool = Field(
        default=True,
        description="Drop epochs that extend outside the recording.",
    )
    experiment_start_event_code: str | None = Field(
        default=None,
        description="Optional boundary event; anchors at/before it are ignored.",
    )
    experiment_end_event_code: str | None = Field(
        default=None,
        description="Optional boundary event; anchors at/after it are ignored.",
    )
    events_source: EventSource = Field(
        default="annotations",
        description="Where input events are read from: annotations, events_tsv, or auto.",
    )
    event_sample_shift_samples: int = Field(
        default=0,
        description="Sample offset applied to event samples before epoch extraction.",
    )
    method: TimeFrequencyMethod = Field(
        default=TimeFrequencyMethod.FIELDTRIP,
        description="Spectral estimation method. Currently only 'fieldtrip' is supported.",
    )

    time_decimation: int = Field(
        default=20,
        ge=1,
        description="Evaluate one TFR time point every N epoch samples.",
    )
    frequency_start_hz: float = Field(
        default=4.0,
        gt=0,
        description="First frequency in the logarithmic MATLAB-like grid.",
    )
    frequency_exponent_max: float = Field(
        default=5.7,
        gt=0,
        description="Maximum exponent in start * 2**exponent.",
    )
    frequency_exponent_step: float = Field(
        default=0.1,
        gt=0,
        description="Exponent step in the logarithmic frequency grid.",
    )
    low_frequency_cutoff_hz: float = Field(
        default=32.0,
        gt=0,
        description="Frequency boundary between low- and high-frequency settings.",
    )
    low_frequency_n_cycles: float = Field(
        default=6.0,
        gt=0,
        description="Number of cycles used for low-frequency time windows.",
    )
    low_frequency_smoothing_fraction: float = Field(
        default=1.0 / 3.0,
        gt=0,
        description="Low-frequency spectral smoothing as a fraction of frequency.",
    )
    high_frequency_window_s: float = Field(
        default=0.1875,
        gt=0,
        description="Fixed high-frequency time window in seconds.",
    )
    high_frequency_smoothing_mode: Literal["adaptive", "fixed"] = Field(
        default="adaptive",
        description="High-frequency smoothing strategy matching the MATLAB scripts.",
    )
    high_frequency_fixed_smoothing_hz: float = Field(
        default=12.0,
        gt=0,
        description="Fixed high-frequency smoothing in Hz when mode='fixed'.",
    )
    min_tapers: int = Field(
        default=1,
        ge=1,
        description="Minimum number of DPSS tapers per frequency.",
    )
    max_tapers: int | None = Field(
        default=None,
        ge=1,
        description="Optional cap on the number of DPSS tapers.",
    )
    fieldtrip_polyorder: int = Field(
        default=0,
        description=(
            "Polynomial order removed before FieldTrip convolution. "
            "0 removes the DC component, matching ft_specest_mtmconvol."
        ),
    )
    fieldtrip_pad_s: float | None = Field(
        default=None,
        gt=0,
        description=(
            "FieldTrip pad duration in seconds. None uses the epoch duration, "
            "matching ft_specest_mtmconvol when pad is omitted."
        ),
    )
    fieldtrip_padtype: Literal["zero"] = Field(
        default="zero",
        description="FieldTrip padding mode. Only zero padding is currently implemented.",
    )

    baseline_window_s: tuple[float, float] = Field(
        default=(-1.3, -0.7),
        description="Baseline window used to compute baseline_db.",
    )
    apply_baseline: bool = Field(
        default=True,
        description=(
            "When true, subtract baseline_db from power_db. baseline_db is "
            "always computed and written."
        ),
    )

    montage_mode: MontageMode = Field(default=MontageMode.MONO)
    bipolar_direction: BipolarDirection = Field(
        default=BipolarDirection.NEXT_MINUS_PREVIOUS
    )
    bipolar_storage: BipolarStorage = Field(default=BipolarStorage.PREVIOUS)
    channels_for_montage: list[str] | str | None = Field(default=None)
    channels_to_exclude_for_montage: list[str] | str | None = Field(default=None)

    @model_validator(mode="after")
    def _validate_windows(self) -> "TimeFrequencyParams":
        if self.tmax_s <= self.tmin_s:
            raise ValueError("tmax_s must be greater than tmin_s.")
        b0, b1 = self.baseline_window_s
        if b1 <= b0:
            raise ValueError("baseline_window_s must be ordered as (start, stop).")
        if self.max_tapers is not None and self.max_tapers < self.min_tapers:
            raise ValueError("max_tapers must be >= min_tapers.")
        return self


class TimeFrequencyWriterParams(BaseWriterParams):
    """Writer parameters for the time-frequency pipeline."""

    pipeline_label: str = "time_frequency"
    output_modality: str = "ieeg"
    output_suffix: str = "tfr"
    output_description: str = "timefrequency"
    output_format: Literal["hdf5", "matlab"] = "hdf5"

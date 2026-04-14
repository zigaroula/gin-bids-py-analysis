"""
Parameter models for the Delphos HFO/spike detection processing.

Contains:
    - DelphosParams: Algorithm parameters for detection
    - DelphosWriterParams: Writer configuration for output files
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from gin_bids_py_analysis.processing.base import BaseProcessingParams, BaseWriterParams
from gin_bids_py_analysis.processing.utils.channels import (
    BipolarDirection,
    BipolarStorage,
    MontageMode,
)


class DelphosParams(BaseProcessingParams):
    """
    Parameters for the Delphos HFO and spike detector.

    The Delphos algorithm uses Derivative of Gaussian (DoG) wavelet transforms
    to detect high-frequency oscillations and spikes in iEEG signals.

    Example::

        params = DelphosParams(
            alpha=0.005,
            detection_type=["Osc", "Spk"],
            freq_band=[[80, 250], [250, 500]],
        )
    """

    # ------------------------------------------------------------------
    # Core detection parameters
    # ------------------------------------------------------------------

    alpha: float = Field(
        default=0.005,
        gt=0.0,
        le=0.1,
        description=(
            "Statistical significance level for automatic thresholding. "
            "Typical values: 0.005 for SEEG, 0.001-0.01 for other modalities."
        ),
    )

    detection_type: list[Literal["Osc", "Spk"]] = Field(
        default=["Osc", "Spk"],
        description=(
            "Types of events to detect. Valid values: 'Osc' (oscillations/HFOs), "
            "'Spk' (spikes)."
        ),
    )

    freq_band: list[list[float]] = Field(
        default_factory=lambda: [[80, 500]],
        description=(
            "Frequency bands for oscillation detection. Each band is a "
            "[low, high] pair in Hz. Common bands: Ripple [80-250], "
            "Fast Ripple [250-500]."
        ),
    )

    # ------------------------------------------------------------------
    # Threshold parameters
    # ------------------------------------------------------------------

    thr_type: str | float = Field(
        default=40.0,
        description=(
            "Threshold type. Use 'auto' for automatic statistical thresholding "
            "based on alpha, or provide a numeric value for fixed thresholding."
        ),
    )

    param_thr: list[float] | None = Field(
        default=None,
        description=(
            "Custom threshold parameters as [hfo_time_thr, hfo_freq_thr, "
            "spike_time_thr, spike_freq_thr]. If None, defaults are used."
        ),
    )

    quantile_method: str = Field(
        default="hazen",
        description=(
            "Method for quantile computation. Options: 'linear', 'lower', 'higher', "
            "'midpoint', 'nearest', 'hazen', 'weibull', 'alpha_beta', "
            "'median_unbiased', 'normal_unbiased'."
        ),
    )

    # ------------------------------------------------------------------
    # Wavelet transform parameters
    # ------------------------------------------------------------------

    nb_voices: int = Field(
        default=12,
        ge=6,
        le=24,
        description="Number of voices per octave for wavelet transform.",
    )

    vanishing_moment: int = Field(
        default=20,
        ge=1,
        le=50,
        description="Vanishing moment parameter for the DoG mother wavelet.",
    )

    # ------------------------------------------------------------------
    # Montage configuration
    # ------------------------------------------------------------------

    montage_mode: MontageMode = Field(
        default=MontageMode.BIPOLAR,
        description="Channel referencing mode applied before detection.",
    )

    bipolar_direction: BipolarDirection = Field(
        default=BipolarDirection.NEXT_MINUS_PREVIOUS,
        description=(
            "Which contact is subtracted from which. "
            "Only used when montage_mode is BIPOLAR."
        ),
    )

    bipolar_storage: BipolarStorage = Field(
        default=BipolarStorage.NEXT_MINUS_PREVIOUS,
        description=(
            "Naming convention for derived bipolar channels. "
            "Only used when montage_mode is BIPOLAR."
        ),
    )

    channels_for_montage: list[str] | str | dict[str, list[str]] | None = Field(
        default=None,
        description=(
            "Channel selector applied before montage and detection. "
            "Pass a list of exact channel names to keep, a regex string "
            "matched via re.fullmatch against each channel name "
            "(e.g. r'[A-Za-z]p?([1-9]|1[0-9])' for one-letter prefix, "
            "optional 'p', index 1-19), or a dict mapping subject IDs to "
            "lists of channel names for per-subject selection. "
            "If None, all channels are used."
        ),
    )
    channels_to_exclude_for_montage: list[str] | str | None = Field(
        default=None,
        description=(
            "Channel selector applied after channels_for_montage and before "
            "montage/detection. Pass a list of exact channel names to remove, "
            "or a regex string matched via re.fullmatch against each channel "
            "name. If None, no channels are excluded."
        ),
    )

    @model_validator(mode="after")
    def _check_params(self) -> "DelphosParams":
        if self.detection_type:
            valid_types = {"Osc", "Spk"}
            invalid = set(self.detection_type) - valid_types
            if invalid:
                raise ValueError(
                    f"Invalid detection_type values: {invalid}. "
                    f"Valid options are: {valid_types}."
                )
        return self


class DelphosWriterParams(BaseWriterParams):
    """
    Writer parameters for Delphos detection output.

    Only ``bids_root`` must be supplied; all routing fields default to
    Delphos-appropriate values.  Use ``output_format`` to choose the output
    backend:

    * ``"hdf5"`` (default) — writes a single ``.h5`` file containing all
      detection data, event counts, rates, and provenance.
    * ``"tsv"`` — writes a BIDS-compatible ``.tsv`` events file.

    Example::

        from pathlib import Path
        params = DelphosWriterParams(
            bids_root=Path("/path/to/bids"),
            output_format="tsv",
        )
    """

    bids_root: Path
    pipeline_label: str = Field(default="delphos")
    output_modality: str = Field(default="ieeg")
    output_description: str = Field(default="delphos")
    output_suffix: str = Field(default="events")
    output_format: Literal["hdf5", "tsv"] = Field(
        default="hdf5",
        description="Output backend: 'hdf5' writes a structured .h5 file; 'tsv' writes a BIDS-compatible events file.",
    )

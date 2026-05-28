"""Utilities for normalizing BrainVision-style annotation metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping
import re

from bidsforge.processing.utils.matlab import matlab_round


_NUMERIC_CODE_RE = re.compile(r"(\d+)\s*$")


@dataclass(frozen=True)
class AnnotationEvent:
    """Normalized event information extracted from an annotation-like source."""

    onset_s: float
    duration_s: float
    event_type: str
    description: str
    code: str | None = None


def parse_annotation_description(description: str) -> tuple[str, str, str | None]:
    """Split an MNE/BrainVision annotation description into type, text, and code."""
    if "/" in description:
        event_type, parsed_description = description.split("/", 1)
    else:
        event_type, parsed_description = "Stimulus", description

    code = _extract_code(event_type, parsed_description)
    return event_type, parsed_description, code


def coerce_annotation_events(
    annotations: Iterable[Mapping[str, Any]] | None,
) -> list[AnnotationEvent]:
    """Convert annotation-like mappings into normalized event records."""
    if not annotations:
        return []

    normalized: list[AnnotationEvent] = []
    for annotation in annotations:
        full_description = str(annotation.get("description", ""))
        event_type, description, code = parse_annotation_description(
            full_description
        )
        normalized.append(
            AnnotationEvent(
                onset_s=float(annotation.get("onset", 0.0)),
                duration_s=float(annotation.get("duration", 0.0)),
                event_type=event_type,
                description=description,
                code=code,
            )
        )

    return normalized


def downsample_events(
    original_events: Any,
    downsampled_fs: float,
    *,
    original_fs: float | None = None,
    event_sample_shift_samples: int = 0,
    event_onset_precision: str = "sample_quantized",
) -> list[dict] | None:
    """Convert source annotations to sample indices at an output sampling rate.

    Args:
        original_events: MNE Annotations object or any list of dicts with keys
                         "onset" (in seconds) and "duration" (in seconds).
        downsampled_fs: Sampling frequency of the exported signal.
        original_fs: Source sampling frequency, required when applying a source
                     sample offset before projection to ``downsampled_fs``.
        event_sample_shift_samples: Offset to apply in source samples before
                                    converting events to the output rate.
        event_onset_precision: ``"sample_quantized"`` snaps to a source sample
                               before downsampling. ``"exact_time"`` keeps
                               exact TSV onsets in seconds and applies the
                               sample shift as a time offset.
                               ``"spm_continuous_sample"`` reproduces SPM's
                               continuous-file event-to-sample conversion after
                               downsampling: source samples are treated as
                               MATLAB/SPM 1-based samples, then converted to
                               BrainVision zero-based marker onsets.

    Returns:
        List of dicts with keys "onset" (in samples), "duration" (in samples),
        "type", and "description", or ``None`` if no events are present.
    """
    if not original_events:
        return None

    if event_sample_shift_samples and original_fs is None:
        raise ValueError(
            "original_fs is required when event_sample_shift_samples is non-zero."
        )
    if event_onset_precision not in {
        "sample_quantized",
        "exact_time",
        "spm_continuous_sample",
    }:
        raise ValueError(
            "event_onset_precision must be 'sample_quantized', 'exact_time', "
            "or 'spm_continuous_sample'."
        )

    downsampled_events = []
    for ann in coerce_annotation_events(original_events):
        description: int | str
        ann_type = ann.event_type
        if ann_type in ("Stimulus", "Response") and ann.code is not None:
            description = int(ann.code)
        else:
            if ann_type in ("Stimulus", "Response"):
                ann_type = "Comment"
            description = ann.description

        if event_onset_precision == "spm_continuous_sample":
            if original_fs is None:
                raise ValueError(
                    "original_fs is required for event_onset_precision="
                    "'spm_continuous_sample'."
                )
            shifted_onset_s = ann.onset_s + (
                (event_sample_shift_samples + 1) / float(original_fs)
            )
            onset_samples = matlab_round(shifted_onset_s * downsampled_fs + 1) - 1
        elif event_onset_precision == "exact_time":
            shifted_onset_s = ann.onset_s
            if event_sample_shift_samples:
                shifted_onset_s += event_sample_shift_samples / float(original_fs)
            onset_samples = matlab_round(shifted_onset_s * downsampled_fs)
        elif event_sample_shift_samples:
            source_onset_sample = (
                matlab_round(ann.onset_s * float(original_fs))
                + event_sample_shift_samples
            )
            onset_samples = matlab_round(
                source_onset_sample * downsampled_fs / float(original_fs)
            )
        else:
            onset_samples = matlab_round(ann.onset_s * downsampled_fs)
        duration_samples = matlab_round(ann.duration_s * downsampled_fs)

        downsampled_events.append(
            {
                "onset": onset_samples,
                "duration": duration_samples,
                "type": ann_type,
                "description": description,
            }
        )

    return downsampled_events if downsampled_events else None


def _extract_code(event_type: str, description: str) -> str | None:
    if event_type in {"Stimulus", "Response"}:
        match = _NUMERIC_CODE_RE.search(description)
        if match:
            return str(int(match.group(1)))

    stripped = description.strip()
    if stripped.isdigit():
        return str(int(stripped))

    return None

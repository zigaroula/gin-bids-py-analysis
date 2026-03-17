"""Utilities for normalizing BrainVision-style annotation metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping
import re


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


def _extract_code(event_type: str, description: str) -> str | None:
    if event_type in {"Stimulus", "Response"}:
        match = _NUMERIC_CODE_RE.search(description)
        if match:
            return str(int(match.group(1)))

    stripped = description.strip()
    if stripped.isdigit():
        return str(int(stripped))

    return None

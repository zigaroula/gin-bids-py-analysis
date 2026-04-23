"""Resolve event inputs for processors that consume continuous recordings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.bids.matching import entities_compatible


EventSource = Literal["annotations", "events_tsv", "auto"]
ResolvedEventSource = Literal["annotations", "events_tsv"]
EventOnsetPrecision = Literal["sample_quantized", "exact_time"]


@dataclass(frozen=True)
class ResolvedInputEvents:
    """Events selected for a continuous-recording processor."""

    events: Any
    source_requested: EventSource
    source_resolved: ResolvedEventSource
    onset_precision: EventOnsetPrecision
    event_file: Path | None = None

    def metadata(self) -> dict[str, str | None]:
        """Return a JSON/HDF5-friendly metadata snapshot."""
        return {
            "events_source_requested": self.source_requested,
            "events_source_resolved": self.source_resolved,
            "events_onset_precision": self.onset_precision,
            "events_file": str(self.event_file) if self.event_file is not None else None,
        }


def resolve_input_events(
    group: BIDSFileGroup,
    raw: Any,
    source: EventSource,
    *,
    primary_file: BIDSFile | None = None,
) -> ResolvedInputEvents:
    """Resolve processor input events from annotations or a secondary TSV.

    ``events_tsv`` intentionally inspects only ``group.secondaries``. This keeps
    grouping explicit and avoids processors silently discovering files on disk.
    """
    if source not in {"annotations", "events_tsv", "auto"}:
        raise ValueError(
            "events_source must be one of 'annotations', 'events_tsv', or 'auto'."
        )

    primary = primary_file or group.primary
    events_file = _matching_events_tsv(group, primary)

    if source == "events_tsv" and events_file is None:
        raise FileNotFoundError(
            f"events_source='events_tsv' requested for {primary.path.name!r}, "
            "but no matching _events.tsv was found in group.secondaries."
        )

    if source == "events_tsv" or (source == "auto" and events_file is not None):
        assert events_file is not None
        return ResolvedInputEvents(
            events=_load_events_tsv_as_annotations(events_file),
            source_requested=source,
            source_resolved="events_tsv",
            onset_precision="exact_time",
            event_file=events_file.path,
        )

    return ResolvedInputEvents(
        events=raw.annotations,
        source_requested=source,
        source_resolved="annotations",
        onset_precision="sample_quantized",
        event_file=None,
    )


def _matching_events_tsv(
    group: BIDSFileGroup,
    primary_file: BIDSFile,
) -> BIDSFile | None:
    candidates = [
        file
        for file in group.secondaries
        if file.suffix == "events"
        and str(file.extension or "").lower() == ".tsv"
        and entities_compatible(primary_file.entities, file.entities)
    ]

    if not candidates:
        return None
    if len(candidates) > 1:
        names = ", ".join(sorted(file.path.name for file in candidates))
        raise ValueError(
            f"Ambiguous _events.tsv secondary for {primary_file.path.name!r}: {names}."
        )
    return candidates[0]


def _load_events_tsv_as_annotations(events_file: BIDSFile) -> list[dict[str, Any]]:
    with events_file.ensure_loaded() as rows:
        return [_events_tsv_row_to_annotation(row, events_file) for row in list(rows)]


def _events_tsv_row_to_annotation(
    row: Mapping[str, Any],
    events_file: BIDSFile,
) -> dict[str, Any]:
    onset_text = _clean_value(row.get("onset"))
    if onset_text is None:
        raise ValueError(f"Missing required 'onset' column in {events_file.path.name!r}.")

    duration_text = _clean_value(row.get("duration")) or "0"
    code = _clean_value(row.get("code"))
    event_type = _event_type_from_row(row)

    if code is not None:
        description = f"{event_type}/S {code}"
    else:
        label = (
            _clean_value(row.get("description"))
            or _clean_value(row.get("trial_type"))
            or "n/a"
        )
        description = label if "/" in label else f"{event_type}/{label}"

    return {
        "onset": float(onset_text),
        "duration": float(duration_text),
        "description": description,
    }


def _event_type_from_row(row: Mapping[str, Any]) -> str:
    raw_event_type = _clean_value(row.get("event_type"))
    if raw_event_type in {"Stimulus", "Response", "Comment"}:
        return raw_event_type

    trial_type = _clean_value(row.get("trial_type"))
    if trial_type in {"Stimulus", "Response", "Comment"}:
        return trial_type

    return "Stimulus"


def _clean_value(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.casefold() in {"n/a", "nan", "none"}:
        return None
    return text

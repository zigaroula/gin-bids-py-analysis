from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bidsforge.processing.base import BaseProcessingResult
from bidsforge.processing.utils.events import coerce_annotation_events
from bidsforge.processing.utils.serialization import OutputTree, compressed


@dataclass
class TimeFrequencyProcessingResult(BaseProcessingResult):
    power_db: np.ndarray = field(default_factory=lambda: np.empty((0, 0, 0, 0), dtype=np.float32))
    baseline_db: np.ndarray = field(default_factory=lambda: np.empty((0, 0, 0), dtype=np.float32))
    trial_ids: list[str] = field(default_factory=list)
    channel_names: list[str] = field(default_factory=list)
    frequency_hz: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    time_s: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.float64))
    original_fs: float = 0.0
    events: list[dict[str, Any]] | None = None

    def to_output_tree(self, *, pipeline_version: str = "unknown") -> OutputTree:
        meta: dict[str, object] = {
            "schema_name": "time_frequency",
            "schema_version": "1.0",
            "sampling_frequency_hz": float(self.original_fs),
            "dimension_order": "trial x channel x frequency x time",
            "baseline_dimension_order": "trial x channel x frequency",
        }
        for key, value in self.metadata.items():
            if value is not None and isinstance(value, (str, int, float, bool, list, tuple)):
                meta[key] = value

        tree: dict[str, Any] = {
            "data": {
                "power_db": compressed(np.asarray(self.power_db, dtype=np.float32)),
                "baseline_db": compressed(np.asarray(self.baseline_db, dtype=np.float32)),
            },
            "axes": {
                "trial_id": np.asarray(self.trial_ids, dtype=object),
                "channel": np.asarray(self.channel_names, dtype=object),
                "frequency_hz": np.asarray(self.frequency_hz, dtype=np.float64),
                "time_s": np.asarray(self.time_s, dtype=np.float64),
            },
            "meta": meta,
            "provenance": {
                "raw_bids_path": str(self.source_group.primary.path),
                "pipeline_name": "time_frequency",
                "pipeline_version": pipeline_version,
            },
        }
        if self.events:
            events = _normalize_events_for_output(self.events)
            tree["events"] = {
                "onset": np.asarray([event["onset"] for event in events], dtype=np.float64),
                "duration": np.asarray(
                    [event["duration"] for event in events],
                    dtype=np.float64,
                ),
                "type": np.asarray([str(event["type"]) for event in events], dtype=object),
                "description": np.asarray(
                    [str(event["description"]) for event in events],
                    dtype=object,
                ),
            }
        return tree


def _normalize_events_for_output(events: Any) -> list[dict[str, Any]]:
    """Return event records with onset, duration, type, and description keys.

    Time-frequency stores source events for provenance. They can come either
    from MNE annotations or from BIDS ``_events.tsv`` rows converted by
    ``resolve_input_events``; TSV-derived rows intentionally have no explicit
    ``type`` key, so normalize through the shared annotation parser before
    serialisation.
    """
    normalized = []
    for event in coerce_annotation_events(events):
        normalized.append(
            {
                "onset": float(event.onset_s),
                "duration": float(event.duration_s),
                "type": event.event_type,
                "description": event.description,
            }
        )
    return normalized

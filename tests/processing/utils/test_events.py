from __future__ import annotations

from mne import Annotations

from bidsforge.processing.utils.events import (
    coerce_annotation_events,
    parse_annotation_description,
)


def test_parse_annotation_description_extracts_brainvision_code() -> None:
    event_type, description, code = parse_annotation_description("Stimulus/S  17")

    assert event_type == "Stimulus"
    assert description == "S  17"
    assert code == "17"


def test_coerce_annotation_events_normalizes_annotations() -> None:
    annotations = Annotations(
        onset=[0.25, 1.0],
        duration=[0.1, 0.0],
        description=["Stimulus/S  10", "Comment/my note"],
    )

    events = coerce_annotation_events(annotations)

    assert [event.code for event in events] == ["10", None]
    assert events[0].onset_s == 0.25
    assert events[0].duration_s == 0.1
    assert events[1].event_type == "Comment"
    assert events[1].description == "my note"




"""Unit tests for extract_anchor_events_with_mne experiment-window filtering."""
from __future__ import annotations

import numpy as np
import pytest
from mne import Annotations, create_info
from mne.io import RawArray

from gin_bids_py_analysis.processing.utils.epoching import extract_anchor_events_with_mne


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_raw(
    n_samples: int,
    sfreq: float,
    onsets: list[float],
    descriptions: list[str],
) -> RawArray:
    info = create_info(ch_names=["CH1"], sfreq=sfreq, ch_types=["seeg"])
    data = np.zeros((1, n_samples), dtype=np.float64)
    raw = RawArray(data, info, verbose="ERROR")
    raw.set_annotations(
        Annotations(
            onset=onsets,
            duration=[0.0] * len(onsets),
            description=descriptions,
        )
    )
    return raw


ANCHOR = "Stimulus/S  10"
START = "Stimulus/S  1"
END = "Stimulus/S  2"
ANCHOR_CODE = "10"
START_CODE = "1"
END_CODE = "2"


# ---------------------------------------------------------------------------
# Baseline — no boundary codes
# ---------------------------------------------------------------------------

def test_no_boundary_returns_all_anchor_events() -> None:
    """All anchor events are returned when no boundary codes are given."""
    raw = _make_raw(
        n_samples=100,
        sfreq=10.0,
        onsets=[1.0, 2.0, 3.0],
        descriptions=[ANCHOR, ANCHOR, ANCHOR],
    )
    events, samples = extract_anchor_events_with_mne(raw, anchor_codes={ANCHOR_CODE})
    assert len(events) == 3
    assert len(samples) == 3
    assert [e.onset_s for e in events] == pytest.approx([1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Start-code filtering
# ---------------------------------------------------------------------------

def test_start_code_excludes_anchors_at_and_before_boundary() -> None:
    """Anchors whose onset is <= the start-code onset are excluded."""
    # start at t=2.0; anchor at t=1.0 (before), t=2.0 (at boundary), t=3.0 (after)
    raw = _make_raw(
        n_samples=60,
        sfreq=10.0,
        onsets=[1.0, 2.0, 2.0, 3.0],
        descriptions=[ANCHOR, START, ANCHOR, ANCHOR],
    )
    events, samples = extract_anchor_events_with_mne(
        raw,
        anchor_codes={ANCHOR_CODE},
        experiment_start_event_code=START_CODE,
    )
    assert len(events) == 1
    assert events[0].onset_s == pytest.approx(3.0)


def test_start_code_missing_from_recording_applies_no_filtering() -> None:
    """When the start code is not found, no filtering occurs."""
    raw = _make_raw(
        n_samples=60,
        sfreq=10.0,
        onsets=[1.0, 2.0, 3.0],
        descriptions=[ANCHOR, ANCHOR, ANCHOR],
    )
    events, _ = extract_anchor_events_with_mne(
        raw,
        anchor_codes={ANCHOR_CODE},
        experiment_start_event_code=START_CODE,  # not present in recording
    )
    assert len(events) == 3


# ---------------------------------------------------------------------------
# End-code filtering
# ---------------------------------------------------------------------------

def test_end_code_excludes_anchors_at_and_after_boundary() -> None:
    """Anchors whose onset is >= the end-code onset are excluded."""
    # end at t=3.0; anchor at t=1.0 (before), t=3.0 (at boundary), t=4.0 (after)
    raw = _make_raw(
        n_samples=70,
        sfreq=10.0,
        onsets=[1.0, 3.0, 3.0, 4.0],
        descriptions=[ANCHOR, END, ANCHOR, ANCHOR],
    )
    events, _ = extract_anchor_events_with_mne(
        raw,
        anchor_codes={ANCHOR_CODE},
        experiment_end_event_code=END_CODE,
    )
    assert len(events) == 1
    assert events[0].onset_s == pytest.approx(1.0)


def test_end_code_missing_from_recording_applies_no_filtering() -> None:
    """When the end code is not found, no filtering occurs."""
    raw = _make_raw(
        n_samples=60,
        sfreq=10.0,
        onsets=[1.0, 2.0, 3.0],
        descriptions=[ANCHOR, ANCHOR, ANCHOR],
    )
    events, _ = extract_anchor_events_with_mne(
        raw,
        anchor_codes={ANCHOR_CODE},
        experiment_end_event_code=END_CODE,
    )
    assert len(events) == 3


# ---------------------------------------------------------------------------
# Combined start + end
# ---------------------------------------------------------------------------

def test_start_and_end_code_keep_only_strictly_interior_anchors() -> None:
    """Only anchors strictly between start and end boundaries are kept."""
    # start at t=1.0, end at t=5.0
    # anchors at: 0.5 (before start), 1.0 (at start), 2.0 (in), 4.0 (in), 5.0 (at end), 6.0 (after end)
    raw = _make_raw(
        n_samples=100,
        sfreq=10.0,
        onsets=[0.5, 1.0, 1.0, 2.0, 4.0, 5.0, 5.0, 6.0],
        descriptions=[ANCHOR, START, ANCHOR, ANCHOR, ANCHOR, END, ANCHOR, ANCHOR],
    )
    events, _ = extract_anchor_events_with_mne(
        raw,
        anchor_codes={ANCHOR_CODE},
        experiment_start_event_code=START_CODE,
        experiment_end_event_code=END_CODE,
    )
    assert len(events) == 2
    assert [e.onset_s for e in events] == pytest.approx([2.0, 4.0])


# ---------------------------------------------------------------------------
# Multiple boundary occurrences
# ---------------------------------------------------------------------------

def test_multiple_start_codes_uses_first_occurrence() -> None:
    """When the start code appears multiple times, the FIRST (earliest) is used."""
    # Two start events at t=1.0 and t=3.0.
    # Using the first (t=1.0) means anchors at t=2.0 and t=4.0 are kept.
    # Using the second (t=3.0) would exclude the anchor at t=2.0.
    raw = _make_raw(
        n_samples=80,
        sfreq=10.0,
        onsets=[1.0, 2.0, 3.0, 4.0],
        descriptions=[START, ANCHOR, START, ANCHOR],
    )
    events, _ = extract_anchor_events_with_mne(
        raw,
        anchor_codes={ANCHOR_CODE},
        experiment_start_event_code=START_CODE,
    )
    assert len(events) == 2
    assert [e.onset_s for e in events] == pytest.approx([2.0, 4.0])


def test_multiple_end_codes_uses_last_occurrence() -> None:
    """When the end code appears multiple times, the LAST (latest) is used."""
    # Two end events at t=2.0 and t=4.0.
    # Using the last (t=4.0) means an anchor at t=3.0 is kept.
    # Using the first (t=2.0) would exclude it.
    raw = _make_raw(
        n_samples=80,
        sfreq=10.0,
        onsets=[1.0, 2.0, 3.0, 4.0],
        descriptions=[ANCHOR, END, ANCHOR, END],
    )
    events, _ = extract_anchor_events_with_mne(
        raw,
        anchor_codes={ANCHOR_CODE},
        experiment_end_event_code=END_CODE,
    )
    assert len(events) == 2
    assert [e.onset_s for e in events] == pytest.approx([1.0, 3.0])


# ---------------------------------------------------------------------------
# Epoch window freedom
# ---------------------------------------------------------------------------

def test_anchor_within_bounds_is_kept_even_when_epoch_would_extend_outside() -> None:
    """Filtering is based on anchor onset only; epoch reach is irrelevant."""
    # start at t=1.0, end at t=5.0.
    # Anchor at t=4.8 with tmax=0.5 would reach t=5.3 > t_end, but the anchor
    # itself is strictly inside, so it is kept.
    raw = _make_raw(
        n_samples=70,
        sfreq=10.0,
        onsets=[1.0, 4.8, 5.0],
        descriptions=[START, ANCHOR, END],
    )
    events, _ = extract_anchor_events_with_mne(
        raw,
        anchor_codes={ANCHOR_CODE},
        experiment_start_event_code=START_CODE,
        experiment_end_event_code=END_CODE,
    )
    assert len(events) == 1
    assert events[0].onset_s == pytest.approx(4.8)


def test_event_sample_shift_offsets_returned_anchor_samples_only() -> None:
    """A sample shift moves epoch anchors without changing reported onset times."""
    raw = _make_raw(
        n_samples=100,
        sfreq=10.0,
        onsets=[1.0, 2.0, 3.0],
        descriptions=[ANCHOR, ANCHOR, ANCHOR],
    )
    events, samples = extract_anchor_events_with_mne(
        raw,
        anchor_codes={ANCHOR_CODE},
        event_sample_shift_samples=-1,
    )

    assert [e.onset_s for e in events] == pytest.approx([1.0, 2.0, 3.0])
    assert samples.tolist() == [9, 19, 29]

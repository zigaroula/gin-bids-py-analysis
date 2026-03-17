from __future__ import annotations

import numpy as np
import pytest

from gin_bids_py_analysis.processing.trial_stats.resolver import ResolvedTrial
from gin_bids_py_analysis.processing.trial_stats.stats import (
    compute_condition_statistics,
    extract_epochs,
)


def _trial(onset_s: float, *, keep: bool = True) -> ResolvedTrial:
    class _DummyFile:
        path = "dummy.vhdr"

    return ResolvedTrial(
        source_file=_DummyFile(),  # type: ignore[arg-type]
        anchor_event_index=0,
        anchor_event_code="10",
        anchor_onset_s=onset_s,
        anchor_duration_s=0.0,
        label="accepted",
        keep=keep,
    )


def test_extract_epochs_drops_partial_trials() -> None:
    data = np.arange(20, dtype=np.float32)[None, :]
    trials = [_trial(0.0), _trial(0.5)]

    extraction = extract_epochs(
        data,
        sfreq=10.0,
        trials=trials,
        tmin_s=-0.1,
        tmax_s=0.1,
        drop_partial_epochs=True,
    )

    assert extraction.epochs.shape == (1, 1, 3)
    assert len(extraction.kept_trials) == 1
    assert extraction.updated_trials[0].keep is False
    assert extraction.updated_trials[0].exclusion_reason == "partial_epoch"


def test_extract_epochs_raises_when_partial_epochs_are_not_dropped() -> None:
    data = np.arange(20, dtype=np.float32)[None, :]

    with pytest.raises(ValueError, match="partial epoch"):
        extract_epochs(
            data,
            sfreq=10.0,
            trials=[_trial(0.0)],
            tmin_s=-0.1,
            tmax_s=0.1,
            drop_partial_epochs=False,
        )


def test_compute_condition_statistics_detects_known_difference() -> None:
    epochs_a = np.full((4, 2, 3), 5.0, dtype=np.float32)
    epochs_b = np.full((4, 2, 3), 1.0, dtype=np.float32)

    t_values, p_values, mean_a, mean_b, mean_difference = compute_condition_statistics(
        epochs_a,
        epochs_b,
        n_channels=2,
        n_times=3,
        equal_var=False,
    )

    assert t_values.shape == (2, 3)
    assert p_values.shape == (2, 3)
    assert np.all(mean_a == 5.0)
    assert np.all(mean_b == 1.0)
    assert np.all(mean_difference == 4.0)
    assert np.all(t_values > 0)

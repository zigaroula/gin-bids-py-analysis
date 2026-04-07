from __future__ import annotations

from pathlib import Path

import numpy as np
from mne import Annotations, create_info
from mne.io import RawArray

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_slope_stats import (
    TrialSlopeStatsParams,
    TrialSlopeStatsProcessing,
)
from gin_bids_py_analysis.processing.utils.trial_resolver import ResolvedTrial


class _MockPyBIDSFile:
    def __init__(self, path: str, entities: dict) -> None:
        self.path = path
        self.entities = entities


def _make_bids_file(path: Path, entities: dict[str, str]) -> BIDSFile:
    return BIDSFile(_MockPyBIDSFile(str(path), entities))


def _make_raw(
    data: np.ndarray,
    ch_names: list[str],
    sfreq: float,
    annotations: Annotations,
) -> RawArray:
    info = create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["seeg"] * len(ch_names))
    raw = RawArray(np.asarray(data, dtype=np.float64), info, verbose="ERROR")
    raw.set_annotations(annotations)
    return raw


class _SlopeResolver:
    def __init__(self, labels: list[str], predictors: list[object]) -> None:
        self._labels = labels
        self._predictors = predictors

    def resolve_trials(self, group, ieeg_file, anchor_events):
        del group
        out = []
        for index, event in enumerate(anchor_events):
            out.append(
                ResolvedTrial(
                    source_file=ieeg_file,
                    anchor_event_index=index,
                    anchor_event_code=event.code,
                    anchor_onset_s=event.onset_s,
                    anchor_duration_s=event.duration_s,
                    label=self._labels[index],
                    metadata={"predictor_value": self._predictors[index]},
                )
            )
        return out


def test_process_group_computes_condition_slopes(tmp_path: Path) -> None:
    ieeg_file = _make_bids_file(
        tmp_path / "sub-01_task-decid_run-1_ieeg.vhdr",
        {
            "subject": "01",
            "task": "decid",
            "run": "1",
            "suffix": "ieeg",
            "extension": ".vhdr",
            "datatype": "ieeg",
        },
    )

    sfreq = 10.0
    ch_names = ["A1"]
    data = np.zeros((1, 100), dtype=np.float32)
    onsets = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    labels = ["accepted", "rejected", "accepted", "rejected", "accepted", "rejected"]
    predictors = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]

    for onset, label, predictor in zip(onsets, labels, predictors):
        start = int(onset * sfreq)
        stop = start + 3
        if label == "accepted":
            value = (2.0 * predictor) + 1.0
        else:
            value = (-1.0 * predictor) + 5.0
        data[0, start:stop] = value

    annotations = Annotations(
        onset=onsets,
        duration=[0.0] * len(onsets),
        description=["Stimulus/S  10"] * len(onsets),
    )
    ieeg_file.attach_data(_make_raw(data, ch_names, sfreq, annotations))

    processor = TrialSlopeStatsProcessing(
        TrialSlopeStatsParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.2,
            condition_a="accepted",
            condition_b="rejected",
            predictor_metadata_key="predictor_value",
            min_trials_per_condition=3,
            p_value_correction_method="none",
        ),
        resolver=_SlopeResolver(labels, predictors),
    )

    result = processor.process_group(BIDSFileGroup(primary=ieeg_file))

    assert result.condition_a_stats_valid is True
    assert result.condition_b_stats_valid is True
    np.testing.assert_allclose(result.condition_a_slope, np.full((1, 3), 2.0), atol=1e-8)
    np.testing.assert_allclose(result.condition_b_slope, np.full((1, 3), -1.0), atol=1e-8)
    assert result.condition_a_trial_count == 3
    assert result.condition_b_trial_count == 3


def test_process_group_excludes_invalid_predictor_trials(tmp_path: Path) -> None:
    ieeg_file = _make_bids_file(
        tmp_path / "sub-01_task-decid_run-1_ieeg.vhdr",
        {
            "subject": "01",
            "task": "decid",
            "run": "1",
            "suffix": "ieeg",
            "extension": ".vhdr",
            "datatype": "ieeg",
        },
    )

    sfreq = 10.0
    ch_names = ["A1"]
    data = np.zeros((1, 100), dtype=np.float32)
    onsets = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    labels = ["accepted", "rejected", "accepted", "rejected", "accepted", "rejected"]
    predictors = [1.0, 1.0, "BAD", 2.0, 3.0, 3.0]

    annotations = Annotations(
        onset=onsets,
        duration=[0.0] * len(onsets),
        description=["Stimulus/S  10"] * len(onsets),
    )
    ieeg_file.attach_data(_make_raw(data, ch_names, sfreq, annotations))

    processor = TrialSlopeStatsProcessing(
        TrialSlopeStatsParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.2,
            condition_a="accepted",
            condition_b="rejected",
            predictor_metadata_key="predictor_value",
            min_trials_per_condition=3,
            p_value_correction_method="none",
        ),
        resolver=_SlopeResolver(labels, predictors),
    )

    result = processor.process_group(BIDSFileGroup(primary=ieeg_file))

    # accepted has one invalid predictor -> 2 usable trials < min_trials=3
    assert result.condition_a_trial_count == 2
    assert result.condition_a_stats_valid is False
    assert any(trial.exclusion_reason == "invalid_predictor_value" for trial in result.resolved_trials)

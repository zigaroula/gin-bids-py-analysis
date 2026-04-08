from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from mne import Annotations, create_info
from mne.io import RawArray

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_slope_stats import (
    TrialSlopeStatsParams,
    TrialSlopeStatsProcessing,
)
from gin_bids_py_analysis.processing.utils.trial_resolver import TableTrialResolver
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
            predictor="predictor_value",
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
            predictor="predictor_value",
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


def test_process_group_experiment_start_code_filters_early_anchors(tmp_path: Path) -> None:
    """Anchors at or before experiment_start_event_code are excluded before slope regression."""
    sfreq = 10.0
    ch_names = ["A1"]
    # anchor at t=1 (before start → excluded), start at t=2,
    # anchors at t=3,4,5,6,7,8 (after start → 3 accepted + 3 rejected)
    data = np.zeros((1, 100), dtype=np.float32)
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
    # Marker "1" = start code, "10" = anchor code
    annotations = Annotations(
        onset=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        duration=[0.0] * 8,
        description=[
            "Stimulus/S  10",  # excluded (before start)
            "Stimulus/S  1",   # start marker
            "Stimulus/S  10",
            "Stimulus/S  10",
            "Stimulus/S  10",
            "Stimulus/S  10",
            "Stimulus/S  10",
            "Stimulus/S  10",
        ],
    )
    ieeg_file.attach_data(_make_raw(data, ch_names, sfreq, annotations))

    # 6 anchors after filtering → labels and predictors for those 6 only
    labels = ["accepted", "rejected", "accepted", "rejected", "accepted", "rejected"]
    predictors = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0]

    processor = TrialSlopeStatsProcessing(
        TrialSlopeStatsParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.2,
            condition_a="accepted",
            condition_b="rejected",
            predictor="predictor_value",
            min_trials_per_condition=3,
            p_value_correction_method="none",
            experiment_start_event_code="1",
        ),
        resolver=_SlopeResolver(labels, predictors),
    )

    result = processor.process_group(BIDSFileGroup(primary=ieeg_file))

    # All 6 filtered anchors must be present; the early one must not appear.
    assert len(result.resolved_trials) == 6
    assert all(t.anchor_onset_s > 2.0 for t in result.resolved_trials)
    assert result.condition_a_trial_count == 3
    assert result.condition_b_trial_count == 3
    assert result.condition_a_stats_valid is True
    assert result.metadata["experiment_start_event_code"] == "1"
    assert result.metadata["experiment_end_event_code"] is None


def test_process_group_activity_zscore_preserves_regression_significance(
    tmp_path: Path,
) -> None:
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
    group = BIDSFileGroup(primary=ieeg_file)

    base_params = dict(
        anchor_event_codes=["10"],
        tmin_s=0.0,
        tmax_s=0.2,
        condition_a="accepted",
        condition_b="rejected",
        predictor="predictor_value",
        min_trials_per_condition=3,
        p_value_correction_method="none",
    )

    raw_result = TrialSlopeStatsProcessing(
        TrialSlopeStatsParams(**base_params),
        resolver=_SlopeResolver(labels, predictors),
    ).process_group(group)
    z_result = TrialSlopeStatsProcessing(
        TrialSlopeStatsParams(
            **base_params,
            activity_scaling="zscore_by_baseline",
            activity_baseline_tmin_s=0.0,
            activity_baseline_tmax_s=0.1,
        ),
        resolver=_SlopeResolver(labels, predictors),
    ).process_group(group)

    np.testing.assert_allclose(z_result.condition_a_r_value, raw_result.condition_a_r_value)
    np.testing.assert_allclose(z_result.condition_b_r_value, raw_result.condition_b_r_value)
    np.testing.assert_allclose(z_result.condition_a_p_value_corrected, raw_result.condition_a_p_value_corrected)
    np.testing.assert_allclose(z_result.condition_b_p_value_corrected, raw_result.condition_b_p_value_corrected)
    assert z_result.activity_scaling == "zscore_by_baseline"
    assert z_result.metadata["activity_scaling"] == "zscore_by_baseline"
    assert z_result.activity_baseline_tmin_s == pytest.approx(0.0)
    assert z_result.activity_baseline_tmax_s == pytest.approx(0.1)
    assert not np.allclose(z_result.condition_a_slope, raw_result.condition_a_slope)
    assert not np.allclose(z_result.condition_a_epoch_means, raw_result.condition_a_epoch_means)


def test_process_group_supports_numeric_condition_rules_with_predictor_extraction(
    tmp_path: Path,
) -> None:
    ieeg_file = _make_bids_file(
        tmp_path / "sub-01_task-rate_run-1_ieeg.vhdr",
        {
            "subject": "01",
            "task": "rate",
            "run": "1",
            "suffix": "ieeg",
            "extension": ".vhdr",
            "datatype": "ieeg",
        },
    )
    beh_path = tmp_path / "sub-01_task-rate_run-1_beh.tsv"
    beh_path.write_text(
        "valence\tscore\n"
        "-1\t1\n"
        "1\t1\n"
        "-2\t2\n"
        "2\t2\n"
        "-3\t3\n"
        "3\t3\n"
        "0\t4\n",
        encoding="utf-8",
    )
    beh_file = _make_bids_file(
        beh_path,
        {
            "subject": "01",
            "task": "rate",
            "run": "1",
            "suffix": "beh",
            "extension": ".tsv",
            "datatype": "beh",
        },
    )

    sfreq = 10.0
    ch_names = ["A1"]
    data = np.zeros((1, 120), dtype=np.float32)
    onsets = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    valence = [-1, 1, -2, 2, -3, 3, 0]
    scores = [1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0]
    for onset, label_value, score in zip(onsets, valence, scores):
        start = int(onset * sfreq)
        stop = start + 3
        if label_value < 0:
            data[0, start:stop] = (2.0 * score) + 1.0
        elif label_value > 0:
            data[0, start:stop] = (-1.0 * score) + 5.0
        else:
            data[0, start:stop] = 9.0

    annotations = Annotations(
        onset=onsets,
        duration=[0.0] * len(onsets),
        description=["Stimulus/S  10"] * len(onsets),
    )
    ieeg_file.attach_data(_make_raw(data, ch_names, sfreq, annotations))

    resolver = TableTrialResolver(
        conditions=[
            {"label": "negative", "when": {"column": "valence", "op": "<", "value": 0}},
            {"label": "positive", "when": {"column": "valence", "op": ">", "value": 0}},
        ],
        extract_columns=["score"],
    )

    result = TrialSlopeStatsProcessing(
        TrialSlopeStatsParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.2,
            condition_a="negative",
            condition_b="positive",
            predictor="score",
            min_trials_per_condition=3,
            p_value_correction_method="none",
        ),
        resolver=resolver,
    ).process_group(BIDSFileGroup(primary=ieeg_file, secondaries=[beh_file]))

    assert result.condition_a_trial_count == 3
    assert result.condition_b_trial_count == 3
    assert result.condition_a_stats_valid is True
    assert result.condition_b_stats_valid is True
    assert any(trial.exclusion_reason == "no_matching_condition" for trial in result.resolved_trials)
    np.testing.assert_allclose(result.condition_a_slope, np.full((1, 3), 2.0), atol=1e-8)
    np.testing.assert_allclose(result.condition_b_slope, np.full((1, 3), -1.0), atol=1e-8)

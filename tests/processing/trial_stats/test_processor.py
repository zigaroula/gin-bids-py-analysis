from __future__ import annotations

from pathlib import Path

import numpy as np
from mne import Annotations

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_stats import (
    ResolvedTrial,
    TrialStatsParams,
    TrialStatsProcessing,
)
import gin_bids_py_analysis.processing.trial_stats.processor as processor_module


class _MockPyBIDSFile:
    def __init__(self, path: str, entities: dict) -> None:
        self.path = path
        self.entities = entities


def _make_bids_file(path: Path, entities: dict[str, str]) -> BIDSFile:
    return BIDSFile(_MockPyBIDSFile(str(path), entities))


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


class _FakeRaw:
    def __init__(
        self,
        data: np.ndarray,
        ch_names: list[str],
        sfreq: float,
        annotations: Annotations,
    ) -> None:
        self._data = data
        self.ch_names = ch_names
        self.info = {"sfreq": sfreq}
        self.annotations = annotations

    def get_data(self) -> np.ndarray:
        return self._data


class _FixedResolver:
    def resolve_trials(self, group, ieeg_file, anchor_events):
        del group
        label_lookup = {
            ("1", 0): "accepted",
            ("1", 1): "rejected",
            ("2", 0): "accepted",
            ("2", 1): "rejected",
        }
        return [
            ResolvedTrial(
                source_file=ieeg_file,
                anchor_event_index=index,
                anchor_event_code=event.code,
                anchor_onset_s=event.onset_s,
                anchor_duration_s=event.duration_s,
                label=label_lookup[(ieeg_file.get("run"), index)],
            )
            for index, event in enumerate(anchor_events)
        ]


class _AlternatingResolver:
    def resolve_trials(self, group, ieeg_file, anchor_events):
        del group
        labels = ["accepted", "rejected", "accepted", "rejected"]
        return [
            ResolvedTrial(
                source_file=ieeg_file,
                anchor_event_index=index,
                anchor_event_code=event.code,
                anchor_onset_s=event.onset_s,
                anchor_duration_s=event.duration_s,
                label=labels[index],
            )
            for index, event in enumerate(anchor_events)
        ]


def test_process_group_pools_multiple_ieeg_files_and_sets_shared_output_entities(
    tmp_path: Path,
) -> None:
    file_run1 = _make_bids_file(
        tmp_path / "sub-01_ses-01_task-decid_run-1_ieeg.vhdr",
        {
            "subject": "01",
            "session": "01",
            "task": "decid",
            "run": "1",
            "suffix": "ieeg",
            "extension": ".vhdr",
            "datatype": "ieeg",
        },
    )
    file_run2 = _make_bids_file(
        tmp_path / "sub-01_ses-01_task-decid_run-2_ieeg.vhdr",
        {
            "subject": "01",
            "session": "01",
            "task": "decid",
            "run": "2",
            "suffix": "ieeg",
            "extension": ".vhdr",
            "datatype": "ieeg",
        },
    )

    sfreq = 10.0
    ch_names = ["A1", "A2"]
    data_run1 = np.zeros((2, 40), dtype=np.float32)
    data_run1[:, 10:13] = 5.0
    data_run1[:, 20:23] = 1.0
    data_run2 = np.zeros((2, 40), dtype=np.float32)
    data_run2[:, 10:13] = 6.0
    data_run2[:, 20:23] = 2.0
    annotations = Annotations(
        onset=[1.0, 2.0],
        duration=[0.0, 0.0],
        description=["Stimulus/S  10", "Stimulus/S  10"],
    )
    file_run1.attach_data(_FakeRaw(data_run1, ch_names, sfreq, annotations))
    file_run2.attach_data(_FakeRaw(data_run2, ch_names, sfreq, annotations))

    processor = TrialStatsProcessing(
        TrialStatsParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.2,
            condition_a="accepted",
            condition_b="rejected",
            min_trials_per_condition=2,
        ),
        resolver=_FixedResolver(),
    )

    result = processor.process_group(
        BIDSFileGroup(primary=file_run1, secondaries=[file_run2])
    )

    assert result.condition_a_trial_count == 2
    assert result.condition_b_trial_count == 2
    assert result.channel_names == ch_names
    assert result.output_entities == {
        "subject": "01",
        "session": "01",
        "task": "decid",
    }
    assert result.stats_valid is True
    assert np.all(result.mean_difference > 0)
    assert len(result.source_ieeg_files) == 2


def test_process_group_aggregates_channels_by_atlas_region(
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
    electrodes_path = tmp_path / "sub-01_task-decid_run-1_electrodes.tsv"
    _write_text(
        electrodes_path,
        "name\tatlasA\nA1\tR1\nA2\tR1\nB1\tR2\n",
    )
    electrodes_file = _make_bids_file(
        electrodes_path,
        {
            "subject": "01",
            "task": "decid",
            "run": "1",
            "suffix": "electrodes",
            "extension": ".tsv",
            "datatype": "ieeg",
        },
    )

    sfreq = 10.0
    ch_names = ["A1", "A2", "B1"]
    data = np.zeros((3, 60), dtype=np.float32)
    data[:, 10:13] = np.array([[6.0], [4.0], [8.0]], dtype=np.float32)
    data[:, 20:23] = np.array([[2.0], [0.0], [4.0]], dtype=np.float32)
    data[:, 30:33] = np.array([[6.0], [4.0], [8.0]], dtype=np.float32)
    data[:, 40:43] = np.array([[2.0], [0.0], [4.0]], dtype=np.float32)
    annotations = Annotations(
        onset=[1.0, 2.0, 3.0, 4.0],
        duration=[0.0, 0.0, 0.0, 0.0],
        description=["Stimulus/S  10"] * 4,
    )
    ieeg_file.attach_data(_FakeRaw(data, ch_names, sfreq, annotations))

    processor = TrialStatsProcessing(
        TrialStatsParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.2,
            condition_a="accepted",
            condition_b="rejected",
            atlas_name="atlasA",
        ),
        resolver=_AlternatingResolver(),
    )
    result = processor.process_group(
        BIDSFileGroup(primary=ieeg_file, secondaries=[electrodes_file])
    )

    assert result.analysis_level == "roi"
    assert result.channel_names == ["R1", "R2"]
    assert result.condition_a_trial_count == 2
    assert result.condition_b_trial_count == 2
    assert result.mean_difference.shape == (2, 3)
    assert np.all(result.mean_difference > 0.0)
    assert result.source_electrodes_files == [str(electrodes_path)]
    assert result.output_entities == {
        "subject": "01",
        "task": "decid",
    }

def test_process_group_drops_na_like_regions_when_atlas_regions_not_set(
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
    electrodes_path = tmp_path / "sub-01_task-decid_run-1_electrodes.tsv"
    _write_text(
        electrodes_path,
        "name\tatlasA\nA1\tR1\nA2\tN/A\nB1\tn.a.\n",
    )
    electrodes_file = _make_bids_file(
        electrodes_path,
        {
            "subject": "01",
            "task": "decid",
            "run": "1",
            "suffix": "electrodes",
            "extension": ".tsv",
            "datatype": "ieeg",
        },
    )

    sfreq = 10.0
    ch_names = ["A1", "A2", "B1"]
    data = np.zeros((3, 60), dtype=np.float32)
    data[:, 10:13] = np.array([[6.0], [4.0], [8.0]], dtype=np.float32)
    data[:, 20:23] = np.array([[2.0], [0.0], [4.0]], dtype=np.float32)
    data[:, 30:33] = np.array([[6.0], [4.0], [8.0]], dtype=np.float32)
    data[:, 40:43] = np.array([[2.0], [0.0], [4.0]], dtype=np.float32)
    annotations = Annotations(
        onset=[1.0, 2.0, 3.0, 4.0],
        duration=[0.0, 0.0, 0.0, 0.0],
        description=["Stimulus/S  10"] * 4,
    )
    ieeg_file.attach_data(_FakeRaw(data, ch_names, sfreq, annotations))

    processor = TrialStatsProcessing(
        TrialStatsParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.2,
            condition_a="accepted",
            condition_b="rejected",
            atlas_name="atlasA",
        ),
        resolver=_AlternatingResolver(),
    )
    result = processor.process_group(
        BIDSFileGroup(primary=ieeg_file, secondaries=[electrodes_file])
    )

    assert result.analysis_level == "roi"
    assert result.channel_names == ["R1"]
    assert result.mean_difference.shape == (1, 3)
    assert np.all(result.mean_difference > 0.0)

def test_process_group_window_ms_binning(
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
    data = np.zeros((1, 60), dtype=np.float32)
    data[:, 10:16] = np.array([[2, 4, 6, 8, 10, 12]], dtype=np.float32)
    data[:, 20:26] = np.array([[1, 1, 1, 1, 1, 1]], dtype=np.float32)
    data[:, 30:36] = np.array([[4, 6, 8, 10, 12, 14]], dtype=np.float32)
    data[:, 40:46] = np.array([[2, 2, 2, 2, 2, 2]], dtype=np.float32)
    annotations = Annotations(
        onset=[1.0, 2.0, 3.0, 4.0],
        duration=[0.0, 0.0, 0.0, 0.0],
        description=["Stimulus/S  10"] * 4,
    )
    ieeg_file.attach_data(_FakeRaw(data, ch_names, sfreq, annotations))

    processor = TrialStatsProcessing(
        TrialStatsParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.5,
            condition_a="accepted",
            condition_b="rejected",
            window_ms=200.0,
        ),
        resolver=_AlternatingResolver(),
    )
    result = processor.process_group(BIDSFileGroup(primary=ieeg_file))

    assert result.analysis_level == "channel"
    assert result.channel_names == ["A1"]
    assert result.window_ms == 200.0
    assert result.n_bins == 0
    assert result.time_axis_s.shape == (3,)
    assert result.mean_difference.shape == (1, 3)
    assert np.all(result.mean_difference > 0.0)


def test_process_group_n_bins_binning(
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
    data = np.zeros((1, 60), dtype=np.float32)
    data[:, 10:16] = np.array([[2, 4, 6, 8, 10, 12]], dtype=np.float32)
    data[:, 20:26] = np.array([[1, 1, 1, 1, 1, 1]], dtype=np.float32)
    data[:, 30:36] = np.array([[4, 6, 8, 10, 12, 14]], dtype=np.float32)
    data[:, 40:46] = np.array([[2, 2, 2, 2, 2, 2]], dtype=np.float32)
    annotations = Annotations(
        onset=[1.0, 2.0, 3.0, 4.0],
        duration=[0.0, 0.0, 0.0, 0.0],
        description=["Stimulus/S  10"] * 4,
    )
    ieeg_file.attach_data(_FakeRaw(data, ch_names, sfreq, annotations))

    processor = TrialStatsProcessing(
        TrialStatsParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.5,
            condition_a="accepted",
            condition_b="rejected",
            n_bins=2,
        ),
        resolver=_AlternatingResolver(),
    )
    result = processor.process_group(BIDSFileGroup(primary=ieeg_file))

    assert result.analysis_level == "channel"
    assert result.channel_names == ["A1"]
    assert result.window_ms == 0.0
    assert result.n_bins == 2
    assert result.time_axis_s.shape == (2,)
    assert result.mean_difference.shape == (1, 2)
    assert np.all(result.mean_difference > 0.0)

def test_temporal_bin_epochs_merges_single_sample_tail() -> None:
    epochs = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], dtype=np.float32)
    time_axis_s = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)

    binned, binned_time = processor_module._temporal_bin_epochs(
        epochs,
        time_axis_s,
        window_samples=2,
    )

    assert binned.shape == (1, 1, 2)
    assert binned_time.shape == (2,)
    np.testing.assert_allclose(
        binned[0, 0, :],
        np.array([1.5, 4.0], dtype=np.float32),
    )
    np.testing.assert_allclose(
        binned_time,
        np.array([-0.75, 0.5], dtype=np.float64),
    )


def test_temporal_bin_epochs_by_n_bins_respects_requested_count() -> None:
    epochs = np.array([[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]], dtype=np.float32)
    time_axis_s = np.array([-0.5, -0.3, -0.1, 0.1, 0.3, 0.5], dtype=np.float64)

    binned, binned_time = processor_module._temporal_bin_epochs_by_n_bins(
        epochs,
        time_axis_s,
        n_bins=2,
    )

    assert binned.shape == (1, 1, 2)
    assert binned_time.shape == (2,)
    np.testing.assert_allclose(
        binned[0, 0, :],
        np.array([2.0, 5.0], dtype=np.float32),
    )

def test_process_group_raises_when_atlas_name_without_matching_electrodes(
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
    annotations = Annotations(
        onset=[1.0, 2.0, 3.0, 4.0],
        duration=[0.0, 0.0, 0.0, 0.0],
        description=["Stimulus/S  10"] * 4,
    )
    ieeg_file.attach_data(_FakeRaw(np.zeros((1, 60), dtype=np.float32), ["A1"], 10.0, annotations))

    processor = TrialStatsProcessing(
        TrialStatsParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.2,
            atlas_name="atlasA",
        ),
        resolver=_AlternatingResolver(),
    )

    try:
        processor.process_group(BIDSFileGroup(primary=ieeg_file))
        assert False, "Expected ValueError for missing electrodes table."
    except ValueError as exc:
        assert "requires a matching *_electrodes.tsv/csv file" in str(exc)


def test_process_group_prefers_least_specific_electrodes_file_on_tie(
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

    generic_path = tmp_path / "sub-01_task-decid_run-1_electrodes.tsv"
    _write_text(generic_path, "name\tatlasA\nA1\tGENERIC\n")
    generic_file = _make_bids_file(
        generic_path,
        {
            "subject": "01",
            "task": "decid",
            "run": "1",
            "suffix": "electrodes",
            "extension": ".tsv",
            "datatype": "ieeg",
        },
    )

    specific_path = tmp_path / "sub-01_task-decid_run-1_space-MNI305_electrodes.tsv"
    _write_text(specific_path, "name\tatlasA\nA1\tSPECIFIC\n")
    specific_file = _make_bids_file(
        specific_path,
        {
            "subject": "01",
            "task": "decid",
            "run": "1",
            "space": "MNI305",
            "suffix": "electrodes",
            "extension": ".tsv",
            "datatype": "ieeg",
        },
    )

    annotations = Annotations(
        onset=[1.0, 2.0, 3.0, 4.0],
        duration=[0.0, 0.0, 0.0, 0.0],
        description=["Stimulus/S  10"] * 4,
    )
    ieeg_file.attach_data(_FakeRaw(np.zeros((1, 60), dtype=np.float32), ["A1"], 10.0, annotations))

    processor = TrialStatsProcessing(
        TrialStatsParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.2,
            atlas_name="atlasA",
        ),
        resolver=_AlternatingResolver(),
    )

    result = processor.process_group(
        BIDSFileGroup(primary=ieeg_file, secondaries=[generic_file, specific_file])
    )
    assert result.channel_names == ["GENERIC"]
    assert result.source_electrodes_files == [str(generic_path)]


def test_process_group_raises_when_electrodes_ambiguity_persists_after_tiebreak(
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

    electrodes_a_path = tmp_path / "sub-01_task-decid_run-1_desc-a_electrodes.tsv"
    _write_text(electrodes_a_path, "name\tatlasA\nA1\tA\n")
    electrodes_a = _make_bids_file(
        electrodes_a_path,
        {
            "subject": "01",
            "task": "decid",
            "run": "1",
            "desc": "a",
            "suffix": "electrodes",
            "extension": ".tsv",
            "datatype": "ieeg",
        },
    )

    electrodes_b_path = tmp_path / "sub-01_task-decid_run-1_desc-b_electrodes.tsv"
    _write_text(electrodes_b_path, "name\tatlasA\nA1\tB\n")
    electrodes_b = _make_bids_file(
        electrodes_b_path,
        {
            "subject": "01",
            "task": "decid",
            "run": "1",
            "desc": "b",
            "suffix": "electrodes",
            "extension": ".tsv",
            "datatype": "ieeg",
        },
    )

    annotations = Annotations(
        onset=[1.0, 2.0, 3.0, 4.0],
        duration=[0.0, 0.0, 0.0, 0.0],
        description=["Stimulus/S  10"] * 4,
    )
    ieeg_file.attach_data(_FakeRaw(np.zeros((1, 60), dtype=np.float32), ["A1"], 10.0, annotations))

    processor = TrialStatsProcessing(
        TrialStatsParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.2,
            atlas_name="atlasA",
        ),
        resolver=_AlternatingResolver(),
    )

    try:
        processor.process_group(
            BIDSFileGroup(primary=ieeg_file, secondaries=[electrodes_a, electrodes_b])
        )
        assert False, "Expected ValueError for persistent electrodes ambiguity."
    except ValueError as exc:
        assert "Ambiguous electrodes table match" in str(exc)





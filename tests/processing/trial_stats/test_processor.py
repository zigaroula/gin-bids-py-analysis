from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from mne import Annotations, create_info
from mne.io import RawArray

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_stats import (
    TableTrialResolver,
    TrialStatsParams,
    TrialStatsProcessing,
)
import gin_bids_py_analysis.processing.utils.epoching as epoching_module
from gin_bids_py_analysis.processing.utils.trial_resolver import ResolvedTrial


class _MockPyBIDSFile:
    def __init__(self, path: str, entities: dict) -> None:
        self.path = path
        self.entities = entities


def _make_bids_file(path: Path, entities: dict[str, str]) -> BIDSFile:
    return BIDSFile(_MockPyBIDSFile(str(path), entities))


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")



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
    file_run1.attach_data(_make_raw(data_run1, ch_names, sfreq, annotations))
    file_run2.attach_data(_make_raw(data_run2, ch_names, sfreq, annotations))

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
    ieeg_file.attach_data(_make_raw(data, ch_names, sfreq, annotations))

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
    ieeg_file.attach_data(_make_raw(data, ch_names, sfreq, annotations))

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
    ieeg_file.attach_data(_make_raw(data, ch_names, sfreq, annotations))

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
    ieeg_file.attach_data(_make_raw(data, ch_names, sfreq, annotations))

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

    binned, binned_time = epoching_module.temporal_bin_epochs(
        epochs,
        time_axis_s,
        window_samples_count=2,
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

    binned, binned_time = epoching_module.temporal_bin_epochs_by_n_bins(
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
    ieeg_file.attach_data(_make_raw(np.zeros((1, 60), dtype=np.float32), ["A1"], 10.0, annotations))

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
    ieeg_file.attach_data(_make_raw(np.zeros((1, 60), dtype=np.float32), ["A1"], 10.0, annotations))

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
    ieeg_file.attach_data(_make_raw(np.zeros((1, 60), dtype=np.float32), ["A1"], 10.0, annotations))

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


# ---------------------------------------------------------------------------
# channel_significance_mode tests
# ---------------------------------------------------------------------------


def _make_ieeg_file_with_data(
    tmp_path: Path,
    data: np.ndarray,
    ch_names: list[str],
    sfreq: float,
    n_events: int,
    event_onset_step_s: float = 1.0,
) -> "BIDSFile":

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
    onsets = [float(i + 1) * event_onset_step_s for i in range(n_events)]
    annotations = Annotations(
        onset=onsets,
        duration=[0.0] * n_events,
        description=["Stimulus/S  10"] * n_events,
    )
    ieeg_file.attach_data(_make_raw(data, ch_names, sfreq, annotations))
    return ieeg_file


def test_process_group_channel_significance_mode_none_gives_null_mask(
    tmp_path: Path,
) -> None:
    sfreq = 10.0
    ch_names = ["A1", "A2"]
    # 4 trials, 2 conditions; channel 0 has a large difference
    data = np.zeros((2, 60), dtype=np.float32)
    data[0, 10:13] = 10.0
    data[0, 30:33] = 10.0
    data[0, 20:23] = 0.0
    data[0, 40:43] = 0.0
    ieeg_file = _make_ieeg_file_with_data(tmp_path, data, ch_names, sfreq, n_events=4)

    processor = TrialStatsProcessing(
        TrialStatsParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.2,
            condition_a="accepted",
            condition_b="rejected",
            channel_significance_mode="none",
        ),
        resolver=_AlternatingResolver(),
    )
    result = processor.process_group(BIDSFileGroup(primary=ieeg_file))

    assert result.channel_significant_mask is None


def test_process_group_channel_significance_mode_single_bin(
    tmp_path: Path,
) -> None:
    sfreq = 10.0
    ch_names = ["A1", "A2"]
    # 4 trials (alternating accepted/rejected), 1 s apart
    # Channel 0: large, consistent difference in accepted vs rejected
    # Channel 1: no difference
    data = np.zeros((2, 60), dtype=np.float32)
    data[0, 10:13] = 10.0  # trial 1 (accepted, anchor=1.0)
    data[0, 20:23] = 0.0   # trial 2 (rejected, anchor=2.0)
    data[0, 30:33] = 10.0  # trial 3 (accepted, anchor=3.0)
    data[0, 40:43] = 0.0   # trial 4 (rejected, anchor=4.0)
    # Channel 1 stays at 0 for all trials

    ieeg_file = _make_ieeg_file_with_data(tmp_path, data, ch_names, sfreq, n_events=4)

    processor = TrialStatsProcessing(
        TrialStatsParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.2,
            condition_a="accepted",
            condition_b="rejected",
            channel_significance_mode="single_bin",
            p_value_correction_method="none",
        ),
        resolver=_AlternatingResolver(),
    )
    result = processor.process_group(BIDSFileGroup(primary=ieeg_file))

    assert result.channel_significant_mask is not None
    assert result.channel_significant_mask.shape == (len(ch_names),)
    assert result.channel_significant_mask.dtype == bool
    assert result.channel_significant_mask[0] is np.bool_(True)
    assert result.channel_significant_mask[1] is np.bool_(False)


def test_process_group_channel_significance_mode_duration(
    tmp_path: Path,
) -> None:
    sfreq = 10.0  # 100 ms per sample
    ch_names = ["A1", "A2"]
    # Use a longer epoch window (1 s = 11 samples at 10 Hz) so that
    # significant bins can accumulate across time.
    # Channel 0: strong consistent difference → multiple significant bins
    # Channel 1: no difference → no significant bins
    data = np.zeros((2, 80), dtype=np.float32)
    # 4 trials, anchors at 1, 2, 3, 4 s; tmin=0, tmax=0.9 → 10 samples
    data[0, 10:20] = 10.0  # trial 1 accepted
    data[0, 20:30] = 0.0   # trial 2 rejected
    data[0, 30:40] = 10.0  # trial 3 accepted
    data[0, 40:50] = 0.0   # trial 4 rejected
    # Channel 1 stays 0

    ieeg_file = _make_ieeg_file_with_data(tmp_path, data, ch_names, sfreq, n_events=4)

    processor = TrialStatsProcessing(
        TrialStatsParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.9,
            condition_a="accepted",
            condition_b="rejected",
            channel_significance_mode="duration",
            channel_significance_duration_threshold_ms=100.0,
            p_value_correction_method="none",
        ),
        resolver=_AlternatingResolver(),
    )
    result = processor.process_group(BIDSFileGroup(primary=ieeg_file))

    assert result.channel_significant_mask is not None
    assert result.channel_significant_mask.shape == (len(ch_names),)
    assert result.channel_significant_mask.dtype == bool
    assert result.channel_significant_mask[0] is np.bool_(True)
    assert result.channel_significant_mask[1] is np.bool_(False)


def test_process_group_experiment_start_code_filters_early_anchors(
    tmp_path: Path,
) -> None:
    """Anchors at or before the start-code onset are excluded from trial resolution."""
    sfreq = 10.0
    ch_names = ["A1"]
    # Events: two anchors before/at start (t=1,2 → excluded), start at t=2 (same sample as t=2
    # anchor → anchor excluded), then four anchors after start (t=3,4,5,6 → 2 accepted + 2 rejected).
    data = np.zeros((1, 80), dtype=np.float32)

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
        onset=[1.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        duration=[0.0] * 7,
        description=[
            "Stimulus/S  10",  # t=1 excluded (before start)
            "Stimulus/S  1",   # t=2 start marker
            "Stimulus/S  10",  # t=2 excluded (at start)
            "Stimulus/S  10",  # t=3 accepted
            "Stimulus/S  10",  # t=4 rejected
            "Stimulus/S  10",  # t=5 accepted
            "Stimulus/S  10",  # t=6 rejected
        ],
    )
    ieeg_file.attach_data(_make_raw(data, ch_names, sfreq, annotations))

    class _ExperimentBoundaryResolver:
        def resolve_trials(self, group, ieeg_file, anchor_events):
            del group
            labels = ["accepted", "rejected", "accepted", "rejected"]
            return [
                ResolvedTrial(
                    source_file=ieeg_file,
                    anchor_event_index=i,
                    anchor_event_code=ev.code,
                    anchor_onset_s=ev.onset_s,
                    anchor_duration_s=ev.duration_s,
                    label=labels[i],
                )
                for i, ev in enumerate(anchor_events)
            ]

    processor = TrialStatsProcessing(
        TrialStatsParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.1,
            condition_a="accepted",
            condition_b="rejected",
            min_trials_per_condition=2,
            p_value_correction_method="none",
            experiment_start_event_code="1",
        ),
        resolver=_ExperimentBoundaryResolver(),
    )

    result = processor.process_group(BIDSFileGroup(primary=ieeg_file))

    # Only the four anchors strictly after the start code should appear.
    assert len(result.resolved_trials) == 4
    assert all(t.anchor_onset_s > 2.0 for t in result.resolved_trials)
    assert result.condition_a_trial_count == 2
    assert result.condition_b_trial_count == 2
    assert result.metadata["experiment_start_event_code"] == "1"
    assert result.metadata["experiment_end_event_code"] is None


def test_process_group_experiment_end_code_filters_late_anchors(
    tmp_path: Path,
) -> None:
    """Anchors at or after the end-code onset are excluded from trial resolution."""
    sfreq = 10.0
    ch_names = ["A1"]
    data = np.zeros((1, 80), dtype=np.float32)

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
    # anchors at t=1 (kept), t=2 (kept), t=3 (kept), t=4 (kept), end at t=5,
    # anchor at t=5 (excluded), t=6 (excluded)
    annotations = Annotations(
        onset=[1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 6.0],
        duration=[0.0] * 7,
        description=[
            "Stimulus/S  10",  # t=1 accepted
            "Stimulus/S  10",  # t=2 rejected
            "Stimulus/S  10",  # t=3 accepted
            "Stimulus/S  10",  # t=4 rejected
            "Stimulus/S  2",   # t=5 end marker
            "Stimulus/S  10",  # t=5 excluded (at end)
            "Stimulus/S  10",  # t=6 excluded (after end)
        ],
    )
    ieeg_file.attach_data(_make_raw(data, ch_names, sfreq, annotations))

    class _ExperimentEndResolver:
        def resolve_trials(self, group, ieeg_file, anchor_events):
            del group
            labels = ["accepted", "rejected", "accepted", "rejected"]
            return [
                ResolvedTrial(
                    source_file=ieeg_file,
                    anchor_event_index=i,
                    anchor_event_code=ev.code,
                    anchor_onset_s=ev.onset_s,
                    anchor_duration_s=ev.duration_s,
                    label=labels[i],
                )
                for i, ev in enumerate(anchor_events)
            ]

    processor = TrialStatsProcessing(
        TrialStatsParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.1,
            condition_a="accepted",
            condition_b="rejected",
            min_trials_per_condition=2,
            p_value_correction_method="none",
            experiment_end_event_code="2",
        ),
        resolver=_ExperimentEndResolver(),
    )

    result = processor.process_group(BIDSFileGroup(primary=ieeg_file))

    assert len(result.resolved_trials) == 4
    assert all(t.anchor_onset_s < 5.0 for t in result.resolved_trials)
    assert result.condition_a_trial_count == 2
    assert result.condition_b_trial_count == 2
    assert result.metadata["experiment_start_event_code"] is None
    assert result.metadata["experiment_end_event_code"] == "2"


def test_process_group_activity_zscore_preserves_ttest_statistics(
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
    data = np.zeros((1, 80), dtype=np.float32)
    data[0, 10:13] = 6.0
    data[0, 20:23] = 1.0
    data[0, 30:33] = 8.0
    data[0, 40:43] = 2.0
    annotations = Annotations(
        onset=[1.0, 2.0, 3.0, 4.0],
        duration=[0.0] * 4,
        description=["Stimulus/S  10"] * 4,
    )
    ieeg_file.attach_data(_make_raw(data, ch_names, sfreq, annotations))
    group = BIDSFileGroup(primary=ieeg_file)

    base_params = dict(
        anchor_event_codes=["10"],
        tmin_s=0.0,
        tmax_s=0.2,
        condition_a="accepted",
        condition_b="rejected",
        min_trials_per_condition=2,
        p_value_correction_method="none",
    )

    raw_result = TrialStatsProcessing(
        TrialStatsParams(**base_params),
        resolver=_AlternatingResolver(),
    ).process_group(group)
    z_result = TrialStatsProcessing(
        TrialStatsParams(
            **base_params,
            activity_scaling="zscore_by_baseline",
            activity_baseline_tmin_s=0.0,
            activity_baseline_tmax_s=0.1,
        ),
        resolver=_AlternatingResolver(),
    ).process_group(group)

    np.testing.assert_allclose(z_result.t_values, raw_result.t_values)
    np.testing.assert_allclose(z_result.p_values, raw_result.p_values)
    np.testing.assert_array_equal(z_result.significant_mask, raw_result.significant_mask)
    assert z_result.activity_scaling == "zscore_by_baseline"
    assert z_result.metadata["activity_scaling"] == "zscore_by_baseline"
    assert z_result.activity_baseline_tmin_s == pytest.approx(0.0)
    assert z_result.activity_baseline_tmax_s == pytest.approx(0.1)
    assert not np.allclose(z_result.mean_difference, raw_result.mean_difference)
    baseline_mask = (z_result.time_axis_s >= 0.0) & (z_result.time_axis_s <= 0.1)
    pooled_baseline = np.concatenate(
        [
            z_result.condition_a_epochs[:, :, baseline_mask],
            z_result.condition_b_epochs[:, :, baseline_mask],
        ],
        axis=0,
    )
    assert abs(float(np.nanmean(pooled_baseline))) < 1e-12


def test_process_group_supports_numeric_condition_rules_and_audits_exclusions(
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
    _write_text(
        beh_path,
        "rating\n-2\n2\n-1\n3\n0\n",
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
    data = np.zeros((1, 80), dtype=np.float32)
    onsets = [1.0, 2.0, 3.0, 4.0, 5.0]
    amplitudes = [6.0, 1.0, 5.0, 2.0, 9.0]
    for onset, amplitude in zip(onsets, amplitudes):
        start = int(onset * sfreq)
        data[0, start : start + 3] = amplitude

    annotations = Annotations(
        onset=onsets,
        duration=[0.0] * len(onsets),
        description=["Stimulus/S  10"] * len(onsets),
    )
    ieeg_file.attach_data(_make_raw(data, ch_names, sfreq, annotations))

    resolver = TableTrialResolver(
        conditions=[
            {"label": "negative", "when": {"column": "rating", "op": "<", "value": 0}},
            {"label": "positive", "when": {"column": "rating", "op": ">", "value": 0}},
        ],
        extract_columns=["rating"],
    )

    result = TrialStatsProcessing(
        TrialStatsParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.2,
            condition_a="negative",
            condition_b="positive",
            min_trials_per_condition=2,
            p_value_correction_method="none",
        ),
        resolver=resolver,
    ).process_group(BIDSFileGroup(primary=ieeg_file, secondaries=[beh_file]))

    assert result.condition_a_trial_count == 2
    assert result.condition_b_trial_count == 2
    assert result.stats_valid is True
    assert np.all(result.mean_difference > 0.0)
    assert any(trial.exclusion_reason == "no_matching_condition" for trial in result.resolved_trials)
    assert any(
        trial.metadata.get("condition_resolution_reason") == "no_matching_condition"
        for trial in result.resolved_trials
    )

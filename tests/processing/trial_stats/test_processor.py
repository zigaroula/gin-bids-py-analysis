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
    def resolve_trials(self, group, hilbert_file, anchor_events):
        del group
        label_lookup = {
            ("1", 0): "accepted",
            ("1", 1): "rejected",
            ("2", 0): "accepted",
            ("2", 1): "rejected",
        }
        return [
            ResolvedTrial(
                source_file=hilbert_file,
                anchor_event_index=index,
                anchor_event_code=event.code,
                anchor_onset_s=event.onset_s,
                anchor_duration_s=event.duration_s,
                label=label_lookup[(hilbert_file.get("run"), index)],
            )
            for index, event in enumerate(anchor_events)
        ]


def test_process_group_pools_multiple_hilbert_files_and_sets_shared_output_entities(
    monkeypatch,
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
    raw_lookup = {
        str(file_run1.path): _FakeRaw(data_run1, ch_names, sfreq, annotations),
        str(file_run2.path): _FakeRaw(data_run2, ch_names, sfreq, annotations),
    }
    monkeypatch.setattr(processor_module, "load_ieeg", lambda file: raw_lookup[str(file.path)])

    processor = TrialStatsProcessing(
        TrialStatsParams(
            anchor_event_codes=["10"],
            tmin_s=0.0,
            tmax_s=0.2,
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
    assert len(result.source_hilbert_files) == 2

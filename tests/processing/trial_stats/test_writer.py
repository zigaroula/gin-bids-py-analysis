from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import scipy.io

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_stats import (
    BaseTrialStatsProcessingResult,
    BaseTrialStatsProcessingWriter,
    BaseTrialStatsWriterParams,
)
from gin_bids_py_analysis.processing.trial_stats.writer import _trial_table_path
from gin_bids_py_analysis.processing.utils.matlab import make_struct, matlab_safe_name
from gin_bids_py_analysis.processing.utils.trial_resolver import ResolvedTrial


class _MockPyBIDSFile:
    def __init__(self, path: str, entities: dict) -> None:
        self.path = path
        self.entities = entities


def _make_bids_file(path: Path, entities: dict[str, str]) -> BIDSFile:
    return BIDSFile(_MockPyBIDSFile(str(path), entities))


def _make_result(tmp_path: Path) -> BaseTrialStatsProcessingResult:
    primary = _make_bids_file(
        tmp_path / "sub-01_task-test_ieeg.vhdr",
        {
            "subject": "01",
            "task": "test",
            "suffix": "ieeg",
            "extension": ".vhdr",
            "datatype": "ieeg",
        },
    )
    resolved_trial = ResolvedTrial(
        source_file=primary,
        anchor_event_index=0,
        anchor_event_code="10",
        anchor_onset_s=1.25,
        anchor_duration_s=0.0,
        label="go/no-go",
        trial_id="trial-01",
        metadata={
            "condition_inputs": {"choice": 1},
            "condition_resolution_reason": "choice == 1",
        },
    )
    data = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    return BaseTrialStatsProcessingResult(
        source_group=BIDSFileGroup(primary=primary),
        output_entities={"subject": "01", "task": "test"},
        metadata={"window_samples": 0, "effective_n_bins": 3},
        time_axis_s=np.array([-0.2, 0.0, 0.2], dtype=np.float64),
        channel_names=["A1"],
        condition_a="go/no-go",
        condition_b="wait stop",
        condition_a_trial_count=1,
        condition_b_trial_count=1,
        condition_a_mean=data,
        condition_b_mean=data + 1.0,
        condition_a_sem=np.full_like(data, 0.1),
        condition_b_sem=np.full_like(data, 0.2),
        sfreq=1000.0,
        resolved_trials=[resolved_trial],
        source_ieeg_files=[str(primary.path)],
        source_table_files=[],
        source_electrodes_files=[],
        analysis_level="channel",
        stats_valid=True,
    )


class _DummyWriter(BaseTrialStatsProcessingWriter):
    def _pipeline_name(self) -> str:
        return "dummy_subject"

    def _write_hdf5_specific(self, fh, result, str_dtype) -> None:
        del result, str_dtype
        stats_grp = fh.create_group("stats")
        stats_grp.create_dataset("value", data=np.array([[1.0]], dtype=np.float64))

    def _build_matlab_specific(self, result):
        del result
        return {
            "stats": make_struct(value=np.array([[1.0]], dtype=np.float64)),
        }


def test_trial_table_path_uses_trials_suffix() -> None:
    output_path = Path("/tmp/sub-01_desc-conditiontest_stats.h5")
    assert _trial_table_path(output_path).name == "sub-01_desc-conditiontest_trials.tsv"


def test_base_writer_writes_shared_hdf5_and_trial_table(tmp_path: Path) -> None:
    writer = _DummyWriter(
        BaseTrialStatsWriterParams(
            bids_root=tmp_path,
            pipeline_label="dummy_subject",
            output_description="dummyoutput",
            output_format="hdf5",
        )
    )
    result = _make_result(tmp_path)

    output_path = writer.write(result)
    trial_table_path = _trial_table_path(output_path)

    assert output_path.name == "sub-01_task-test_desc-dummyoutput_stats.h5"
    assert trial_table_path.exists()

    with h5py.File(output_path, "r") as fh:
        assert fh["meta"]["binning_mode"].asstr()[()] == "none"
        assert fh["meta"]["stats_valid"][()] == 1
        assert fh["provenance"]["pipeline_name"].asstr()[()] == "dummy_subject"
        assert fh["trials"]["condition_inputs"].asstr()[0] == '{"choice": 1}'


def test_base_writer_writes_shared_matlab_with_safe_condition_names(tmp_path: Path) -> None:
    writer = _DummyWriter(
        BaseTrialStatsWriterParams(
            bids_root=tmp_path,
            pipeline_label="dummy_subject",
            output_description="dummyoutput",
            output_format="matlab",
        )
    )
    result = _make_result(tmp_path)

    output_path = writer.write(result)
    data = scipy.io.loadmat(
        str(output_path),
        squeeze_me=True,
        struct_as_record=False,
    )["data"]

    assert output_path.suffix == ".mat"
    assert hasattr(data.means, matlab_safe_name(result.condition_a))
    assert hasattr(data.means, matlab_safe_name(result.condition_b))
    assert str(data.provenance.pipeline_name) == "dummy_subject"

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import scipy.io

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_slope_stats import (
    TrialSlopeStatsProcessingResult,
    TrialSlopeStatsProcessingWriter,
    TrialSlopeStatsWriterParams,
    load_trial_slope_stats_result,
)
from gin_bids_py_analysis.processing.utils.trial_resolver import ResolvedTrial


class _MockPyBIDSFile:
    def __init__(self, path: str, entities: dict) -> None:
        self.path = path
        self.entities = entities


def _make_bids_file(path: Path, entities: dict[str, str]) -> BIDSFile:
    return BIDSFile(_MockPyBIDSFile(str(path), entities))


def _make_result(tmp_path: Path) -> TrialSlopeStatsProcessingResult:
    primary = _make_bids_file(
        tmp_path / "sub-01_task-decid_ieeg.vhdr",
        {
            "subject": "01",
            "task": "decid",
            "suffix": "ieeg",
            "extension": ".vhdr",
            "datatype": "ieeg",
        },
    )
    shape = (2, 3)
    return TrialSlopeStatsProcessingResult(
        source_group=BIDSFileGroup(primary=primary),
        output_entities={"subject": "01", "task": "decid"},
        condition_a_slope=np.full(shape, 1.0, dtype=np.float64),
        condition_a_intercept=np.full(shape, 2.0, dtype=np.float64),
        condition_a_r_value=np.full(shape, 0.8, dtype=np.float64),
        condition_a_p_value=np.full(shape, 0.01, dtype=np.float64),
        condition_a_p_value_corrected=np.full(shape, 0.02, dtype=np.float64),
        condition_a_significant_mask=np.ones(shape, dtype=bool),
        condition_b_slope=np.full(shape, -1.0, dtype=np.float64),
        condition_b_intercept=np.full(shape, 3.0, dtype=np.float64),
        condition_b_r_value=np.full(shape, -0.75, dtype=np.float64),
        condition_b_p_value=np.full(shape, 0.03, dtype=np.float64),
        condition_b_p_value_corrected=np.full(shape, 0.04, dtype=np.float64),
        condition_b_significant_mask=np.zeros(shape, dtype=bool),
        condition_a_mean=np.full(shape, 4.0, dtype=np.float64),
        condition_b_mean=np.full(shape, 5.0, dtype=np.float64),
        condition_a_sem=np.full(shape, 0.4, dtype=np.float64),
        condition_b_sem=np.full(shape, 0.5, dtype=np.float64),
        time_axis_s=np.array([0.0, 0.1, 0.2], dtype=np.float64),
        channel_names=["A1", "A2"],
        condition_a="accepted",
        condition_b="rejected",
        condition_a_trial_count=3,
        condition_b_trial_count=3,
        condition_a_trials_used=3,
        condition_b_trials_used=3,
        sfreq=10.0,
        condition_a_predictor_values=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        condition_b_predictor_values=np.array([1.0, 2.0, 3.0], dtype=np.float64),
        resolved_trials=[
            ResolvedTrial(
                source_file=primary,
                anchor_event_index=0,
                anchor_event_code="10",
                anchor_onset_s=1.0,
                anchor_duration_s=0.0,
                label="accepted",
                keep=True,
                metadata={"predictor_raw": "1.0", "predictor_value": 1.0},
            )
        ],
        source_ieeg_files=[str(primary.path)],
        source_table_files=[str(tmp_path / "beh.tsv")],
        source_electrodes_files=[],
        analysis_level="channel",
        analysis_type="slope_regression",
        predictor_metadata_key="predictor_value",
        predictor_scaling="none",
        p_value_correction_method="fdr_bh",
        significance_alpha=0.05,
        condition_a_stats_valid=True,
        condition_b_stats_valid=True,
        stats_valid=True,
    )


def test_writer_outputs_hdf5_and_loader_roundtrip(tmp_path: Path) -> None:
    result = _make_result(tmp_path)
    writer = TrialSlopeStatsProcessingWriter(
        TrialSlopeStatsWriterParams(
            bids_root=tmp_path,
            output_description="trialslopestats",
            output_format="hdf5",
        )
    )

    output_path = writer.write(result)
    assert output_path.exists()

    with h5py.File(output_path, "r") as fh:
        assert fh["meta"]["analysis_type"].asstr()[()] == "slope_regression"
        assert fh["regression"]["condition_a"]["slope"].shape == (2, 3)
        assert fh["regression"]["condition_b"]["p_value_corrected"].shape == (2, 3)
        assert fh["predictor"]["condition_a_values"].shape == (3,)
        assert fh["predictor"]["condition_b_values"].shape == (3,)
        assert list(fh["trials"]["predictor_raw"].asstr()[:]) == ["1.0"]

    loaded = load_trial_slope_stats_result(output_path)
    np.testing.assert_allclose(loaded.condition_a_slope, result.condition_a_slope)
    np.testing.assert_allclose(loaded.condition_b_p_value_corrected, result.condition_b_p_value_corrected)
    assert loaded.predictor_metadata_key == "predictor_value"


def test_writer_outputs_matlab(tmp_path: Path) -> None:
    result = _make_result(tmp_path)
    writer = TrialSlopeStatsProcessingWriter(
        TrialSlopeStatsWriterParams(
            bids_root=tmp_path,
            output_description="trialslopestats",
            output_format="matlab",
        )
    )
    output_path = writer.write(result)
    assert output_path.suffix == ".mat"
    mat = scipy.io.loadmat(str(output_path), squeeze_me=True, struct_as_record=False)
    data = mat["data"]
    assert str(data.meta.analysis_type) == "slope_regression"
    assert data.regression.condition_a.slope.shape == (2, 3)

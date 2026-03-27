from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import h5py
import numpy as np
import scipy.io

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_stats_group import (
    ROIChannelContribution,
    TrialStatsGroupProcessingResult,
    TrialStatsGroupProcessingWriter,
    TrialStatsGroupWriterParams,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


class _MockPyBIDSFile:
    def __init__(self, path: str, entities: dict) -> None:
        self.path = path
        self.entities = entities


def _make_case_dir(case_name: str) -> Path:
    root = REPO_ROOT / "tests" / "_script_tmp"
    root.mkdir(parents=True, exist_ok=True)
    case_dir = root / f"{case_name}_{uuid.uuid4().hex[:8]}"
    case_dir.mkdir(parents=True, exist_ok=False)
    return case_dir


def _make_bids_file(path: Path, entities: dict[str, str]) -> BIDSFile:
    return BIDSFile(_MockPyBIDSFile(str(path), entities))


def test_writer_outputs_expected_hdf5_schema_and_group_path() -> None:
    case_dir = _make_case_dir("writer_schema")
    try:
        primary = _make_bids_file(
            case_dir / "sub-01_task-decid_desc-trialstats_stats.h5",
            {
                "subject": "01",
                "task": "decid",
                "desc": "trialstats",
                "suffix": "stats",
                "extension": ".h5",
                "datatype": "ieeg",
            },
        )

        result = TrialStatsGroupProcessingResult(
            source_group=BIDSFileGroup(primary=primary),
            output_entities={"subject": "group", "task": "decid"},
            t_values=np.array([[2.0, 3.0]], dtype=np.float64),
            p_values=np.array([[0.02, 0.03]], dtype=np.float64),
            p_values_uncorrected=np.array([[0.02, 0.03]], dtype=np.float64),
            significant_mask=np.array([[True, True]], dtype=bool),
            metric_mean=np.array([[1.0, 1.5]], dtype=np.float64),
            metric_sem=np.array([[0.2, 0.3]], dtype=np.float64),
            time_axis_s=np.array([0.0, 0.1], dtype=np.float64),
            region_names=["ROI_A"],
            source_metric="mean_difference",
            condition_labels=("accepted", "rejected"),
            p_value_correction_method="none",
            significance_alpha=0.05,
            roi_mode="manual",
            atlas_name=None,
            epoch_mean_t_values=np.array([3.4], dtype=np.float64),
            epoch_mean_p_values=np.array([0.01], dtype=np.float64),
            epoch_mean_df=np.array([4.0], dtype=np.float64),
            epoch_mean_metric_mean=np.array([1.2], dtype=np.float64),
            epoch_mean_metric_sem=np.array([0.25], dtype=np.float64),
            roi_channel_counts=np.array([5], dtype=np.int64),
            roi_subject_counts=np.array([3], dtype=np.int64),
            contributions=[
                ROIChannelContribution(
                    roi="ROI_A",
                    subject="01",
                    channel="A1",
                    source_stats_file=str(primary.path),
                )
            ],
            condition_a_group_mean=np.array([[2.0, 2.5]], dtype=np.float64),
            condition_a_group_sem=np.array([[0.1, 0.2]], dtype=np.float64),
            condition_b_group_mean=np.array([[1.0, 1.2]], dtype=np.float64),
            condition_b_group_sem=np.array([[0.05, 0.1]], dtype=np.float64),
            source_trial_stats_files=[str(primary.path)],
            source_electrodes_files=[],
            excluded_rois={"ROI_B": "insufficient_subjects:1<2"},
        )

        writer = TrialStatsGroupProcessingWriter(
            TrialStatsGroupWriterParams(bids_root=case_dir)
        )
        out_path = writer.write(result)

        assert out_path.exists()
        assert "sub-group" in str(out_path)
        assert "trial_stats_group" in str(out_path)
        assert out_path.name == "sub-group_task-decid_desc-trialstatsgroup_stats.h5"

        with h5py.File(out_path, "r") as fh:
            assert fh["stats"]["t_values"].shape == (1, 2)
            assert fh["means"]["metric_mean"].shape == (1, 2)
            assert fh["means"]["condition_a_mean"].shape == (1, 2)
            assert fh["means"]["condition_a_sem"].shape == (1, 2)
            assert fh["means"]["condition_b_mean"].shape == (1, 2)
            assert fh["means"]["condition_b_sem"].shape == (1, 2)
            assert fh["uncertainty"]["metric_sem"].shape == (1, 2)
            assert fh["summary_epoch"]["t_values"].shape == (1,)
            assert list(fh["axes"]["region"].asstr()[:]) == ["ROI_A"]
            assert fh["meta"]["analysis_level"].asstr()[()] == "roi_group"
            assert fh["meta"]["source_metric"].asstr()[()] == "mean_difference"
            assert fh["meta"]["roi_mode"].asstr()[()] == "manual"
            assert list(fh["meta"]["excluded_rois"]["region"].asstr()[:]) == ["ROI_B"]
            assert list(fh["contributions"]["channel"].asstr()[:]) == ["A1"]
            assert list(fh["provenance"]["source_trial_stats_files"].asstr()[:]) == [str(primary.path)]
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_writer_outputs_matlab_format() -> None:
    case_dir = _make_case_dir("writer_matlab")
    try:
        primary = _make_bids_file(
            case_dir / "sub-01_task-decid_desc-trialstats_stats.h5",
            {
                "subject": "01",
                "task": "decid",
                "desc": "trialstats",
                "suffix": "stats",
                "extension": ".h5",
                "datatype": "ieeg",
            },
        )

        result = TrialStatsGroupProcessingResult(
            source_group=BIDSFileGroup(primary=primary),
            output_entities={"subject": "group", "task": "decid"},
            t_values=np.array([[2.0, 3.0]], dtype=np.float64),
            p_values=np.array([[0.02, 0.03]], dtype=np.float64),
            p_values_uncorrected=np.array([[0.02, 0.03]], dtype=np.float64),
            significant_mask=np.array([[True, True]], dtype=bool),
            metric_mean=np.array([[1.0, 1.5]], dtype=np.float64),
            metric_sem=np.array([[0.2, 0.3]], dtype=np.float64),
            time_axis_s=np.array([0.0, 0.1], dtype=np.float64),
            region_names=["ROI_A"],
            source_metric="mean_difference",
            condition_labels=("accepted", "rejected"),
            p_value_correction_method="none",
            significance_alpha=0.05,
            roi_mode="manual",
            atlas_name=None,
            epoch_mean_t_values=np.array([3.4], dtype=np.float64),
            epoch_mean_p_values=np.array([0.01], dtype=np.float64),
            epoch_mean_df=np.array([4.0], dtype=np.float64),
            epoch_mean_metric_mean=np.array([1.2], dtype=np.float64),
            epoch_mean_metric_sem=np.array([0.25], dtype=np.float64),
            roi_channel_counts=np.array([5], dtype=np.int64),
            roi_subject_counts=np.array([3], dtype=np.int64),
            contributions=[
                ROIChannelContribution(
                    roi="ROI_A",
                    subject="01",
                    channel="A1",
                    source_stats_file=str(primary.path),
                )
            ],
            condition_a_group_mean=np.array([[2.0, 2.5]], dtype=np.float64),
            condition_a_group_sem=np.array([[0.1, 0.2]], dtype=np.float64),
            condition_b_group_mean=np.array([[1.0, 1.2]], dtype=np.float64),
            condition_b_group_sem=np.array([[0.05, 0.1]], dtype=np.float64),
            source_trial_stats_files=[str(primary.path)],
            source_electrodes_files=[],
            excluded_rois={"ROI_B": "insufficient_subjects:1<2"},
        )

        writer = TrialStatsGroupProcessingWriter(
            TrialStatsGroupWriterParams(
                bids_root=case_dir,
                output_format="matlab",
            )
        )
        out_path = writer.write(result)

        assert out_path.suffix == ".mat"
        assert out_path.exists()
        assert "sub-group" in str(out_path)
        assert "trial_stats_group" in str(out_path)
        assert out_path.name == "sub-group_task-decid_desc-trialstatsgroup_stats.mat"

        mat = scipy.io.loadmat(str(out_path), squeeze_me=True, struct_as_record=False)
        data = mat["data"]
        # With squeeze_me=True, (1, 2) arrays become (2,) and (1,) become scalars
        assert np.atleast_1d(data.stats.t_values).shape == (2,)
        assert np.atleast_1d(data.stats.p_values).shape == (2,)
        assert np.atleast_1d(data.stats.significant_mask).shape == (2,)
        assert np.atleast_1d(data.means.metric_mean).shape == (2,)
        assert np.atleast_1d(data.means.condition_a_mean).shape == (2,)
        assert np.atleast_1d(data.means.condition_b_mean).shape == (2,)
        assert np.atleast_1d(data.means.condition_a_sem).shape == (2,)
        assert np.atleast_1d(data.means.condition_b_sem).shape == (2,)
        assert np.atleast_1d(data.uncertainty.metric_sem).shape == (2,)
        assert np.atleast_1d(data.summary_epoch.t_values).shape == (1,)
        assert int(np.atleast_1d(data.summary_epoch.roi_channel_counts)[0]) == 5
        region_arr = np.atleast_1d(data.axes.region)
        assert list(region_arr) == ["ROI_A"]
        np.testing.assert_allclose(np.atleast_1d(data.axes.time_s), [0.0, 0.1])
        assert str(data.meta.analysis_level) == "roi_group"
        assert str(data.meta.source_metric) == "mean_difference"
        assert str(data.meta.roi_mode) == "manual"
        channel_arr = np.atleast_1d(data.contributions.channel)
        assert list(channel_arr) == ["A1"]
        assert str(data.provenance.pipeline_name) == "trial_stats_group"
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)

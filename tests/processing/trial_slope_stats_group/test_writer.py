from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import h5py
import numpy as np

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_slope_stats_group import (
    ROIChannelContribution,
    TrialSlopeStatsGroupProcessingResult,
    TrialSlopeStatsGroupProcessingWriter,
    TrialSlopeStatsGroupWriterParams,
    load_trial_slope_stats_group_result,
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


def _make_result(primary: BIDSFile, n_rois: int = 1, n_t: int = 2) -> TrialSlopeStatsGroupProcessingResult:
    shape = (n_rois, n_t)
    return TrialSlopeStatsGroupProcessingResult(
        source_group=BIDSFileGroup(primary=primary),
        output_entities={"subject": "group", "task": "decid"},
        metadata={
            "p_value_correction_method": "none",
            "significance_alpha": 0.05,
            "roi_mode": "manual",
            "atlas_name": None,
            "binning_mode": "none",
            "window_ms": 0.0,
            "n_bins": 0,
            "effective_n_bins": n_t,
        },
        condition_a_slope_t_values=np.full(shape, 2.0, dtype=np.float64),
        condition_a_slope_p_values=np.full(shape, 0.02, dtype=np.float64),
        condition_a_slope_p_values_uncorrected=np.full(shape, 0.02, dtype=np.float64),
        condition_a_slope_significant_mask=np.ones(shape, dtype=bool),
        condition_a_slope_mean=np.full(shape, 1.5, dtype=np.float64),
        condition_a_slope_sem=np.full(shape, 0.3, dtype=np.float64),
        condition_a_epoch_slope_t=np.full(n_rois, 3.0, dtype=np.float64),
        condition_a_epoch_slope_p=np.full(n_rois, 0.01, dtype=np.float64),
        condition_a_epoch_slope_df=np.full(n_rois, 4.0, dtype=np.float64),
        condition_a_epoch_slope_mean=np.full(n_rois, 1.5, dtype=np.float64),
        condition_a_epoch_slope_sem=np.full(n_rois, 0.25, dtype=np.float64),
        condition_b_slope_t_values=np.full(shape, -2.0, dtype=np.float64),
        condition_b_slope_p_values=np.full(shape, 0.04, dtype=np.float64),
        condition_b_slope_p_values_uncorrected=np.full(shape, 0.04, dtype=np.float64),
        condition_b_slope_significant_mask=np.zeros(shape, dtype=bool),
        condition_b_slope_mean=np.full(shape, -1.5, dtype=np.float64),
        condition_b_slope_sem=np.full(shape, 0.3, dtype=np.float64),
        condition_b_epoch_slope_t=np.full(n_rois, -2.5, dtype=np.float64),
        condition_b_epoch_slope_p=np.full(n_rois, 0.05, dtype=np.float64),
        condition_b_epoch_slope_df=np.full(n_rois, 4.0, dtype=np.float64),
        condition_b_epoch_slope_mean=np.full(n_rois, -1.5, dtype=np.float64),
        condition_b_epoch_slope_sem=np.full(n_rois, 0.25, dtype=np.float64),
        condition_a_activity_mean=np.full(shape, 2.0, dtype=np.float64),
        condition_a_activity_sem=np.full(shape, 0.2, dtype=np.float64),
        condition_b_activity_mean=np.full(shape, 1.0, dtype=np.float64),
        condition_b_activity_sem=np.full(shape, 0.15, dtype=np.float64),
        condition_a_r_value_mean=np.full(shape, 0.6, dtype=np.float64),
        condition_a_r_value_sem=np.full(shape, 0.05, dtype=np.float64),
        condition_b_r_value_mean=np.full(shape, -0.5, dtype=np.float64),
        condition_b_r_value_sem=np.full(shape, 0.04, dtype=np.float64),
        time_axis_s=np.arange(n_t) * 0.1,
        region_names=[f"ROI_{i}" for i in range(n_rois)],
        condition_labels=("accepted", "rejected"),
        roi_channel_counts=np.full(n_rois, 2, dtype=np.int64),
        roi_subject_counts=np.full(n_rois, 2, dtype=np.int64),
        contributions=[
            ROIChannelContribution(roi="ROI_0", subject="01", channel="A1", source_stats_file=str(primary.path)),
            ROIChannelContribution(roi="ROI_0", subject="02", channel="A1", source_stats_file=str(primary.path)),
        ],
        condition_a_slope_contributions=[np.ones((2, n_t), dtype=np.float64) * 1.5 for _ in range(n_rois)],
        condition_b_slope_contributions=[np.ones((2, n_t), dtype=np.float64) * -1.5 for _ in range(n_rois)],
        condition_a_activity_contributions=[np.ones((2, n_t), dtype=np.float64) * 2.0 for _ in range(n_rois)],
        condition_b_activity_contributions=[np.ones((2, n_t), dtype=np.float64) for _ in range(n_rois)],
        contribution_labels=[["01/A1", "02/A1"] for _ in range(n_rois)],
        p_value_correction_method="none",
        significance_alpha=0.05,
        roi_mode="manual",
        atlas_name=None,
        source_trial_slope_stats_files=[str(primary.path)],
        source_electrodes_files=[],
        excluded_rois={"ROI_BAD": "no_channels"},
    )


def test_writer_hdf5_schema_and_path() -> None:
    case_dir = _make_case_dir("writer_schema")
    try:
        primary = _make_bids_file(
            case_dir / "sub-01_task-decid_desc-slopestat_stats.h5",
            {"subject": "01", "task": "decid", "desc": "slopestat", "suffix": "stats", "extension": ".h5"},
        )
        result = _make_result(primary)
        writer = TrialSlopeStatsGroupProcessingWriter(
            TrialSlopeStatsGroupWriterParams(bids_root=case_dir)
        )
        out_path = writer.write(result)

        assert out_path.exists()
        assert "sub-group" in out_path.name
        assert "trial_slope_stats_group" in str(out_path)
        assert out_path.suffix == ".h5"

        with h5py.File(out_path, "r") as fh:
            assert "regression" in fh
            assert "condition_a" in fh["regression"]
            assert "condition_b" in fh["regression"]
            assert "t_values" in fh["regression"]["condition_a"]
            assert "p_values" in fh["regression"]["condition_a"]
            assert "significant_mask" in fh["regression"]["condition_a"]
            assert "slope_mean" in fh["regression"]["condition_a"]
            assert "epoch_summary" in fh["regression"]["condition_a"]
            assert "means" in fh
            assert "condition_a_mean" in fh["means"]
            assert "condition_b_mean" in fh["means"]
            assert "r_values" in fh
            assert "condition_a_mean" in fh["r_values"]
            assert "axes" in fh
            assert "region" in fh["axes"]
            assert "time_s" in fh["axes"]
            assert "meta" in fh
            assert "contributions" in fh
            assert "roi" in fh["contributions"]
            assert "contribution_samples" in fh
            assert "provenance" in fh

            assert fh["regression"]["condition_a"]["t_values"].shape == (1, 2)
            assert fh["regression"]["condition_a"]["epoch_summary"]["t"].shape == (1,)
            assert list(fh["axes"]["region"].asstr()[:]) == ["ROI_0"]
            assert fh["meta"]["roi_mode"].asstr()[()] == "manual"
            assert list(fh["excluded_rois"]["name"].asstr()[:]) == ["ROI_BAD"]
            assert list(fh["contributions"]["channel"].asstr()[:]) == ["A1", "A1"]
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_writer_hdf5_roundtrip_via_loader() -> None:
    case_dir = _make_case_dir("writer_roundtrip")
    try:
        primary = _make_bids_file(
            case_dir / "sub-01_task-decid_desc-slopestat_stats.h5",
            {"subject": "01", "task": "decid", "desc": "slopestat", "suffix": "stats", "extension": ".h5"},
        )
        result = _make_result(primary, n_rois=2, n_t=3)
        writer = TrialSlopeStatsGroupProcessingWriter(
            TrialSlopeStatsGroupWriterParams(bids_root=case_dir)
        )
        out_path = writer.write(result)
        loaded = load_trial_slope_stats_group_result(out_path)

        assert loaded.region_names == result.region_names
        assert len(loaded.time_axis_s) == 3
        np.testing.assert_allclose(loaded.condition_a_slope_t_values, result.condition_a_slope_t_values)
        np.testing.assert_allclose(loaded.condition_b_slope_t_values, result.condition_b_slope_t_values)
        np.testing.assert_allclose(loaded.condition_a_activity_mean, result.condition_a_activity_mean)
        np.testing.assert_allclose(loaded.condition_b_r_value_mean, result.condition_b_r_value_mean)
        assert loaded.p_value_correction_method == "none"
        assert loaded.roi_mode == "manual"
        assert loaded.excluded_rois == {"ROI_BAD": "no_channels"}
        assert len(loaded.contributions) == 2
        assert len(loaded.condition_a_slope_contributions) == 2
        assert loaded.condition_a_slope_contributions[0].shape == (2, 3)
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)

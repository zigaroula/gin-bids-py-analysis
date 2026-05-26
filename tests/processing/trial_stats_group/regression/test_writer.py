from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import h5py
import numpy as np

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_stats_group import (
    ROIChannelContribution,
    RegressionGroupProcessingResult,
    RegressionGroupProcessingWriter,
    RegressionGroupWriterParams,
    load_regression_group_result,
)
from gin_bids_py_analysis.processing.trial_stats_group.regression.result import (
    RegressionMetricStats,
    ScatterData,
    VsZeroStatsPair,
)
from gin_bids_py_analysis.processing.trial_stats_group.result import (
    GroupEpochStats,
    GroupEstimate,
    GroupEstimatePair,
    GroupTimecourseStats,
    IndexedConditionContributions,
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


def _make_result(primary: BIDSFile, n_rois: int = 1, n_t: int = 2) -> RegressionGroupProcessingResult:
    shape = (n_rois, n_t)
    return RegressionGroupProcessingResult(
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
            "activity_zscore": "none",
        },
        regression_stats=RegressionMetricStats(
            contrast=GroupTimecourseStats(
                t_values=np.full(shape, 2.0, dtype=np.float64),
                p_values=np.full(shape, 0.02, dtype=np.float64),
                p_values_uncorrected=np.full(shape, 0.02, dtype=np.float64),
                significant_mask=np.ones(shape, dtype=bool),
            ),
            epoch_summary=GroupEpochStats(
                t=np.full(n_rois, 3.0, dtype=np.float64),
                p=np.full(n_rois, 0.01, dtype=np.float64),
                df=np.full(n_rois, 4.0, dtype=np.float64),
            ),
            vs_zero=VsZeroStatsPair(
                condition_a=GroupTimecourseStats(
                    t_values=np.full(shape, 1.8, dtype=np.float64),
                    p_values=np.full(shape, 0.04, dtype=np.float64),
                    p_values_uncorrected=np.full(shape, 0.04, dtype=np.float64),
                    significant_mask=np.ones(shape, dtype=bool),
                ),
                condition_b=GroupTimecourseStats(
                    t_values=np.full(shape, -1.2, dtype=np.float64),
                    p_values=np.full(shape, 0.08, dtype=np.float64),
                    p_values_uncorrected=np.full(shape, 0.08, dtype=np.float64),
                    significant_mask=np.zeros(shape, dtype=bool),
                ),
            ),
        ),
        signal_activity_stats=GroupTimecourseStats(
            t_values=np.full(shape, 1.5, dtype=np.float64),
            p_values=np.full(shape, 0.05, dtype=np.float64),
            p_values_uncorrected=np.full(shape, 0.05, dtype=np.float64),
            significant_mask=np.zeros(shape, dtype=bool),
        ),
        signal_activity_epoch=GroupEpochStats(
            t=np.full(n_rois, 1.2, dtype=np.float64),
            p=np.full(n_rois, 0.1, dtype=np.float64),
            df=np.full(n_rois, 3.0, dtype=np.float64),
        ),
        slope=GroupEstimatePair(
            condition_a=GroupEstimate(
                mean=np.full(shape, 1.5, dtype=np.float64),
                sem=np.full(shape, 0.3, dtype=np.float64),
            ),
            condition_b=GroupEstimate(
                mean=np.full(shape, -1.5, dtype=np.float64),
                sem=np.full(shape, 0.3, dtype=np.float64),
            ),
        ),
        signal_activity=GroupEstimatePair(
            condition_a=GroupEstimate(
                mean=np.full(shape, 2.0, dtype=np.float64),
                sem=np.full(shape, 0.2, dtype=np.float64),
            ),
            condition_b=GroupEstimate(
                mean=np.full(shape, 1.0, dtype=np.float64),
                sem=np.full(shape, 0.15, dtype=np.float64),
            ),
        ),
        r_value=GroupEstimatePair(
            condition_a=GroupEstimate(
                mean=np.full(shape, 0.6, dtype=np.float64),
                sem=np.full(shape, 0.05, dtype=np.float64),
            ),
            condition_b=GroupEstimate(
                mean=np.full(shape, -0.5, dtype=np.float64),
                sem=np.full(shape, 0.04, dtype=np.float64),
            ),
        ),
        time_axis_s=np.arange(n_t) * 0.1,
        region_names=[f"ROI_{i}" for i in range(n_rois)],
        condition_labels=("accepted", "rejected"),
        roi_channel_counts=np.full(n_rois, 2, dtype=np.int64),
        roi_subject_counts=np.full(n_rois, 2, dtype=np.int64),
        contributions=[
            ROIChannelContribution(roi="ROI_0", subject="01", channel="A1", source_stats_file=str(primary.path)),
            ROIChannelContribution(roi="ROI_0", subject="02", channel="A1", source_stats_file=str(primary.path)),
        ],
        slope_contributions=IndexedConditionContributions(
            condition_a=[
                np.ones((2, n_t), dtype=np.float64) * 1.5 for _ in range(n_rois)
            ],
            condition_b=[
                np.ones((2, n_t), dtype=np.float64) * -1.5 for _ in range(n_rois)
            ],
            labels=[["01/A1", "02/A1"] for _ in range(n_rois)],
        ),
        signal_activity_contributions=IndexedConditionContributions(
            condition_a=[
                np.ones((2, n_t), dtype=np.float64) * 2.0 for _ in range(n_rois)
            ],
            condition_b=[np.ones((2, n_t), dtype=np.float64) for _ in range(n_rois)],
            labels=[["01/A1", "02/A1"] for _ in range(n_rois)],
        ),
        primary_regression_metric="slope",
        contrast_mode="paired",
        p_value_correction_method="none",
        significance_alpha=0.05,
        roi_mode="manual",
        atlas_name=None,
        source_subject_stats_files=[str(primary.path)],
        source_electrodes_files=[],
        excluded_rois={"ROI_BAD": "no_channels"},
        scatter=ScatterData(),
    )


def test_writer_hdf5_schema_and_path() -> None:
    case_dir = _make_case_dir("writer_schema")
    try:
        primary = _make_bids_file(
            case_dir / "sub-01_task-decid_desc-slopestat_stats.h5",
            {"subject": "01", "task": "decid", "desc": "slopestat", "suffix": "stats", "extension": ".h5"},
        )
        result = _make_result(primary)
        writer = RegressionGroupProcessingWriter(
            RegressionGroupWriterParams(bids_root=case_dir)
        )
        out_path = writer.write(result)

        assert out_path.exists()
        assert "sub-group" in out_path.name
        assert "regression_group" in str(out_path)
        assert out_path.suffix == ".h5"

        with h5py.File(out_path, "r") as fh:
            assert "regression" in fh["stats"]
            assert "condition_contrast" in fh["stats"]["regression"]
            assert "t_values" in fh["stats"]["regression"]["condition_contrast"]
            assert "p_values" in fh["stats"]["regression"]["condition_contrast"]
            assert "significant_mask" in fh["stats"]["regression"]["condition_contrast"]
            assert "accepted" in fh["data"]["regression"]
            assert "slope" in fh["data"]["regression"]["accepted"]
            assert "r_value" in fh["data"]["regression"]["accepted"]
            assert "epoch_summary" in fh["stats"]["regression"]["condition_contrast"]
            assert "signal_activity" in fh["stats"]
            assert "t_values" in fh["stats"]["signal_activity"]
            assert "p_values" in fh["stats"]["signal_activity"]
            assert "epoch_summary" in fh["stats"]["signal_activity"]
            assert "signal_activity" in fh["data"]
            assert "accepted" in fh["data"]["signal_activity"]
            assert "rejected" in fh["data"]["signal_activity"]
            assert "axes" in fh
            assert "region" in fh["axes"]
            assert "time_s" in fh["axes"]
            assert "meta" in fh
            assert "contributions" in fh
            assert "summary" in fh["contributions"]
            assert "signal_activity" in fh["contributions"]
            assert "regression" in fh["contributions"]
            assert "provenance" in fh

            assert fh["stats"]["regression"]["condition_contrast"]["t_values"].shape == (1, 2)
            assert fh["stats"]["regression"]["condition_contrast"]["epoch_summary"]["t"].shape == (1,)
            assert list(fh["axes"]["region"].asstr()[:]) == ["ROI_0"]
            assert fh["meta"]["roi_mode"].asstr()[()] == "manual"
            assert list(fh["excluded_rois"]["region"].asstr()[:]) == ["ROI_BAD"]
            assert list(fh["contributions"]["summary"]["channel"].asstr()[:]) == ["A1", "A1"]
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
        writer = RegressionGroupProcessingWriter(
            RegressionGroupWriterParams(bids_root=case_dir)
        )
        out_path = writer.write(result)
        loaded = load_regression_group_result(out_path)

        assert loaded.region_names == result.region_names
        assert len(loaded.time_axis_s) == 3
        np.testing.assert_allclose(loaded.regression_stats.contrast.t_values, result.regression_stats.contrast.t_values)
        np.testing.assert_allclose(
            loaded.signal_activity_stats.t_values,
            result.signal_activity_stats.t_values,
        )
        np.testing.assert_allclose(loaded.signal_activity.condition_a.mean, result.signal_activity.condition_a.mean)
        np.testing.assert_allclose(loaded.r_value.condition_b.mean, result.r_value.condition_b.mean)
        assert loaded.p_value_correction_method == "none"
        assert loaded.roi_mode == "manual"
        assert loaded.excluded_rois == {"ROI_BAD": "no_channels"}
        assert len(loaded.contributions) == 2
        assert len(loaded.slope_contributions.condition_a) == 2
        assert loaded.slope_contributions.condition_a[0].shape == (2, 3)
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)






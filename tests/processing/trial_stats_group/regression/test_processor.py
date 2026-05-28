from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

import h5py
import numpy as np
import pytest

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_stats_group import (
    RegressionGroupParams,
    RegressionGroupProcessing,
    build_regression_compatible_groups,
)
from gin_bids_py_analysis.processing.trial_stats.regression import load_regression_result
from gin_bids_py_analysis.processing.utils.serialization import write_matlab_tree


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


def _write_slope_stats_h5(
    path: Path,
    *,
    analysis_level: str = "channel",
    channels: list[str],
    time_s: np.ndarray,
    condition_a_slope: np.ndarray,
    condition_b_slope: np.ndarray,
    condition_a_mean: np.ndarray | None = None,
    condition_b_mean: np.ndarray | None = None,
    condition_a_r_value: np.ndarray | None = None,
    condition_b_r_value: np.ndarray | None = None,
    condition_labels: tuple[str, str] = ("accepted", "rejected"),
    predictor: str = "predictor_value",
    predictor_zscore: str = "none",
    predictor_transform_by_condition: dict[str, dict[str, float]] | None = None,
    activity_zscore: str = "none",
    activity_baseline_tmin_s: float = -0.2,
    activity_baseline_tmax_s: float = 0.0,
    source_ieeg_files: list[str] | None = None,
    source_electrodes_files: list[str] | None = None,
) -> None:
    str_dtype = h5py.string_dtype(encoding="utf-8")
    n_ch = len(channels)
    n_t = len(time_s)

    if condition_a_mean is None:
        condition_a_mean = np.ones((n_ch, n_t), dtype=np.float64)
    if condition_b_mean is None:
        condition_b_mean = np.ones((n_ch, n_t), dtype=np.float64)
    if condition_a_r_value is None:
        condition_a_r_value = np.full((n_ch, n_t), 0.5, dtype=np.float64)
    if condition_b_r_value is None:
        condition_b_r_value = np.full((n_ch, n_t), 0.4, dtype=np.float64)

    label_a, label_b = condition_labels

    with h5py.File(path, "w") as fh:
        stats = fh.create_group("stats")
        reg = stats.create_group("regression")
        ca = reg.create_group(label_a)
        ca.create_dataset("slope", data=condition_a_slope.astype(np.float64))
        ca.create_dataset("r_value", data=condition_a_r_value.astype(np.float64))
        ca.create_dataset("p_value", data=np.full((n_ch, n_t), 0.01, dtype=np.float64))
        ca.create_dataset("p_value_corrected", data=np.full((n_ch, n_t), 0.01, dtype=np.float64))
        ca.create_dataset("significant_mask", data=np.ones((n_ch, n_t), dtype=bool))
        cb = reg.create_group(label_b)
        cb.create_dataset("slope", data=condition_b_slope.astype(np.float64))
        cb.create_dataset("r_value", data=condition_b_r_value.astype(np.float64))
        cb.create_dataset("p_value", data=np.full((n_ch, n_t), 0.05, dtype=np.float64))
        cb.create_dataset("p_value_corrected", data=np.full((n_ch, n_t), 0.05, dtype=np.float64))
        cb.create_dataset("significant_mask", data=np.zeros((n_ch, n_t), dtype=bool))

        data = fh.create_group("data")
        signal_activity = data.create_group("signal_activity")
        sa_a = signal_activity.create_group(label_a)
        sa_a.create_dataset("mean", data=condition_a_mean.astype(np.float64))
        sa_a.create_dataset("sem", data=np.full((n_ch, n_t), 0.1, dtype=np.float64))
        sa_b = signal_activity.create_group(label_b)
        sa_b.create_dataset("mean", data=condition_b_mean.astype(np.float64))
        sa_b.create_dataset("sem", data=np.full((n_ch, n_t), 0.1, dtype=np.float64))

        predictor_group = fh.create_group("predictor")
        for label in (label_a, label_b):
            pred_cond = predictor_group.create_group(label)
            pred_cond.create_dataset("raw_values", data=np.array([], dtype=np.float64))
            pred_cond.create_dataset("transformed_values", data=np.array([], dtype=np.float64))
            pred_cond.create_dataset("values", data=np.array([], dtype=np.float64))

        axes = fh.create_group("axes")
        axis_name = "region" if analysis_level != "channel" else "channel"
        axes.create_dataset(axis_name, data=np.array(channels, dtype=object), dtype=str_dtype)
        axes.create_dataset("time_s", data=time_s.astype(np.float64))

        meta = fh.create_group("meta")
        meta.create_dataset("schema_version", data="3.0", dtype=str_dtype)
        meta.create_dataset("analysis_type", data="slope_regression", dtype=str_dtype)
        meta.create_dataset("analysis_level", data=analysis_level, dtype=str_dtype)
        meta.create_dataset("condition_labels", data=np.array(list(condition_labels), dtype=object), dtype=str_dtype)
        meta.create_dataset("available_regression_metrics", data=np.array(["slope", "r_value"], dtype=object), dtype=str_dtype)
        meta.create_dataset("trial_counts", data=np.array([0, 0], dtype=np.int64))
        meta.create_dataset("sampling_frequency_hz", data=0.0)
        meta.create_dataset("p_value_correction_method", data="fdr_bh", dtype=str_dtype)
        meta.create_dataset("significance_alpha", data=0.05)
        meta.create_dataset("stats_valid", data=True)
        meta.create_dataset("binning_mode", data="none", dtype=str_dtype)
        meta.create_dataset("window_ms", data=0.0)
        meta.create_dataset("n_bins", data=0)
        meta.create_dataset("effective_n_bins", data=n_t)
        meta.create_dataset("predictor", data=predictor, dtype=str_dtype)
        meta.create_dataset("predictor_zscore", data=predictor_zscore, dtype=str_dtype)
        meta.create_dataset(
            "predictor_transform_by_condition_json",
            data=json.dumps(
                predictor_transform_by_condition or {},
                sort_keys=True,
                separators=(",", ":"),
            ),
            dtype=str_dtype,
        )
        meta.create_dataset("activity_zscore", data=activity_zscore, dtype=str_dtype)
        meta.create_dataset("activity_baseline_tmin_s", data=activity_baseline_tmin_s)
        meta.create_dataset("activity_baseline_tmax_s", data=activity_baseline_tmax_s)
        meta.create_dataset("activity_baseline_scope", data="global", dtype=str_dtype)
        meta.create_dataset("activity_baseline_remove_outlier_trial_means", data=False)
        meta.create_dataset("trial_activity_summary_kind", data="epoch_mean", dtype=str_dtype)
        meta.create_dataset("trial_activity_summary_missing_response_policy", data="nan_if_missing", dtype=str_dtype)
        meta.create_dataset("trial_activity_summary_source_json", data="{}", dtype=str_dtype)
        meta.create_dataset("trial_activity_summary_label", data="Epoch mean activity", dtype=str_dtype)
        meta.create_dataset("epoch_cleaning_json", data="{}", dtype=str_dtype)
        meta.create_dataset("epoch_cleaning_audit_json", data="{}", dtype=str_dtype)

        prov = fh.create_group("provenance")
        prov.create_dataset(
            "source_ieeg_files",
            data=np.array(source_ieeg_files or [], dtype=object),
            dtype=str_dtype,
        )
        prov.create_dataset(
            "source_electrodes_files",
            data=np.array(source_electrodes_files or [], dtype=object),
            dtype=str_dtype,
        )


def test_build_compatible_groups_splits_heterogeneous_inputs() -> None:
    case_dir = _make_case_dir("groups_split")
    try:
        time_s = np.array([0.0, 0.1, 0.2], dtype=np.float64)
        slope = np.ones((2, 3), dtype=np.float64)

        path_a = case_dir / "sub-01_task-decid_desc-slopestat_stats.h5"
        path_b = case_dir / "sub-02_task-decid_desc-slopestat_stats.h5"
        path_c = case_dir / "sub-03_task-other_desc-slopestat_stats.h5"

        for path in (path_a, path_b, path_c):
            _write_slope_stats_h5(path, channels=["A1", "A2"], time_s=time_s, condition_a_slope=slope, condition_b_slope=slope)

        files = [
            _make_bids_file(path_a, {"subject": "01", "task": "decid", "desc": "slopestat", "suffix": "stats", "extension": ".h5"}),
            _make_bids_file(path_b, {"subject": "02", "task": "decid", "desc": "slopestat", "suffix": "stats", "extension": ".h5"}),
            _make_bids_file(path_c, {"subject": "03", "task": "other", "desc": "slopestat", "suffix": "stats", "extension": ".h5"}),
        ]

        groups = build_regression_compatible_groups(files)
        assert len(groups) == 2
        assert sorted(len(g.all_files) for g in groups) == [1, 2]
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_build_compatible_groups_splits_activity_zscore_inputs() -> None:
    case_dir = _make_case_dir("groups_split_activity_zscore")
    try:
        time_s = np.array([0.0, 0.1, 0.2], dtype=np.float64)
        slope = np.ones((2, 3), dtype=np.float64)

        path_a = case_dir / "sub-01_task-decid_desc-slopestat_stats.h5"
        path_b = case_dir / "sub-02_task-decid_desc-slopestat_stats.h5"
        _write_slope_stats_h5(path_a, channels=["A1", "A2"], time_s=time_s, condition_a_slope=slope, condition_b_slope=slope)
        _write_slope_stats_h5(
            path_b,
            channels=["A1", "A2"],
            time_s=time_s,
            condition_a_slope=slope,
            condition_b_slope=slope,
            activity_zscore="baseline",
        )

        files = [
            _make_bids_file(path_a, {"subject": "01", "task": "decid", "desc": "slopestat", "suffix": "stats", "extension": ".h5"}),
            _make_bids_file(path_b, {"subject": "02", "task": "decid", "desc": "slopestat", "suffix": "stats", "extension": ".h5"}),
        ]

        groups = build_regression_compatible_groups(files)
        assert len(groups) == 2
        assert all(len(group.all_files) == 1 for group in groups)
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_load_regression_result_from_matlab_schema_reports_available_metrics() -> None:
    case_dir = _make_case_dir("matlab_schema")
    try:
        path = case_dir / "sub-01_desc-onset_stats.mat"
        time_s = np.array([0.0, 0.1, 0.2], dtype=np.float64)
        slope_a = np.ones((2, 3), dtype=np.float64)
        slope_b = np.ones((2, 3), dtype=np.float64) * -1
        r_a = np.ones((2, 3), dtype=np.float64) * 0.5
        r_b = np.ones((2, 3), dtype=np.float64) * -0.5
        mean_a = np.ones((2, 3), dtype=np.float64) * 2
        mean_b = np.ones((2, 3), dtype=np.float64) * 3

        write_matlab_tree(
            path,
            {
                "stats": {
                    "regression": {
                        "pleasant": {"slope": slope_a, "r_value": r_a},
                        "unpleasant": {"slope": slope_b, "r_value": r_b},
                    }
                },
                "data": {
                    "signal_activity": {
                        "pleasant": {"mean": mean_a},
                        "unpleasant": {"mean": mean_b},
                    }
                },
                "predictor": {
                    "pleasant": {
                        "raw_values": np.array([], dtype=np.float64),
                        "transformed_values": np.array([], dtype=np.float64),
                        "values": np.array([], dtype=np.float64),
                    },
                    "unpleasant": {
                        "raw_values": np.array([], dtype=np.float64),
                        "transformed_values": np.array([], dtype=np.float64),
                        "values": np.array([], dtype=np.float64),
                    },
                },
                "axes": {
                    "channel": np.array(["A1", "A2"], dtype=object),
                    "time_s": time_s,
                },
                "meta": {
                    "schema_version": "3.0",
                    "analysis_type": "slope_regression",
                    "analysis_level": "channel",
                    "condition_labels": np.array(["pleasant", "unpleasant"], dtype=object),
                    "available_regression_metrics": np.array(["slope", "r_value"], dtype=object),
                    "binning_mode": "none",
                    "window_ms": 0.0,
                    "n_bins": 0,
                    "effective_n_bins": len(time_s),
                    "sampling_frequency_hz": 0.0,
                    "p_value_correction_method": "fdr_bh",
                    "significance_alpha": 0.05,
                    "stats_valid": True,
                    "predictor": "matlab_zscore",
                    "predictor_zscore": "none",
                    "predictor_transform_by_condition_json": "{}",
                    "activity_zscore": "none",
                    "activity_baseline_tmin_s": -0.2,
                    "activity_baseline_tmax_s": 0.0,
                    "activity_baseline_scope": "global",
                    "activity_baseline_remove_outlier_trial_means": False,
                    "trial_activity_summary_kind": "epoch_mean",
                    "trial_activity_summary_missing_response_policy": "nan_if_missing",
                    "trial_activity_summary_source_json": "{}",
                    "trial_activity_summary_label": "Epoch mean activity",
                    "epoch_cleaning_json": "{}",
                    "epoch_cleaning_audit_json": "{}",
                },
                "provenance": {
                    "source_ieeg_files": np.array([], dtype=object),
                    "source_electrodes_files": np.array([], dtype=object),
                },
            },
            root_name="regression",
        )

        result = load_regression_result(path)

        assert (result.condition_a, result.condition_b) == ("pleasant", "unpleasant")
        assert set(result.metadata["available_regression_metrics"]) == {"slope", "r_value"}
        np.testing.assert_allclose(result.regression.condition_a.slope, slope_a)
        np.testing.assert_allclose(result.signal_activity.condition_b.mean, mean_b)
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_process_group_rejects_non_channel_inputs() -> None:
    case_dir = _make_case_dir("reject_non_channel")
    try:
        path = case_dir / "sub-01_task-decid_desc-slopestat_stats.h5"
        _write_slope_stats_h5(
            path,
            analysis_level="roi",
            channels=["ROI1"],
            time_s=np.array([0.0, 0.1]),
            condition_a_slope=np.ones((1, 2)),
            condition_b_slope=np.ones((1, 2)),
        )
        file = _make_bids_file(path, {"subject": "01", "task": "decid", "desc": "slopestat", "suffix": "stats", "extension": ".h5"})
        processor = RegressionGroupProcessing(
            RegressionGroupParams(
                roi_mode="manual",
                manual_region_channels={"ROI1": {"01": ["ROI1"]}},
            )
        )
        with pytest.raises(ValueError, match="channel-level"):
            processor.process_group(BIDSFileGroup(primary=file))
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_process_group_manual_mode_shapes_and_values() -> None:
    case_dir = _make_case_dir("manual_shapes")
    try:
        time_s = np.array([0.0, 0.1, 0.2], dtype=np.float64)

        # Sub-01: A1 slope_a = [2, 2, 2]; B1 slope_a = [-1, -1, -1]
        path_01 = case_dir / "sub-01_task-decid_desc-slopestat_stats.h5"
        slope_a_01 = np.array([[2.0, 2.0, 2.0], [1.0, 1.0, 1.0], [-1.0, -1.0, -1.0]], dtype=np.float64)
        slope_b_01 = np.array([[-2.0, -2.0, -2.0], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=np.float64)
        _write_slope_stats_h5(path_01, channels=["A1", "A2", "B1"], time_s=time_s, condition_a_slope=slope_a_01, condition_b_slope=slope_b_01)

        # Sub-02: A1 slope_a = [3, 3, 3]; B1 slope_a = [0.5, 0.5, 0.5]
        path_02 = case_dir / "sub-02_task-decid_desc-slopestat_stats.h5"
        slope_a_02 = np.array([[3.0, 3.0, 3.0], [0.5, 0.5, 0.5]], dtype=np.float64)
        slope_b_02 = np.array([[-3.0, -3.0, -3.0], [-0.5, -0.5, -0.5]], dtype=np.float64)
        _write_slope_stats_h5(path_02, channels=["A1", "B1"], time_s=time_s, condition_a_slope=slope_a_02, condition_b_slope=slope_b_02)

        file_01 = _make_bids_file(path_01, {"subject": "01", "task": "decid", "desc": "slopestat", "suffix": "stats", "extension": ".h5"})
        file_02 = _make_bids_file(path_02, {"subject": "02", "task": "decid", "desc": "slopestat", "suffix": "stats", "extension": ".h5"})

        processor = RegressionGroupProcessing(
            RegressionGroupParams(
                roi_mode="manual",
                manual_region_channels={
                    "ROI_POS": {"01": ["A1", "A2"], "02": ["A1"]},
                    "ROI_MIX": {"01": ["B1"], "02": ["B1"]},
                },
            )
        )
        result = processor.process_group(BIDSFileGroup(primary=file_01, secondaries=[file_02]))

        assert result.region_names == ["ROI_POS", "ROI_MIX"]
        n_rois = 2
        n_t = 3
        assert result.regression_stats.contrast.t_values.shape == (n_rois, n_t)
        assert result.regression_stats.contrast.p_values.shape == (n_rois, n_t)
        assert result.signal_activity_stats.t_values.shape == (n_rois, n_t)
        assert result.signal_activity.condition_a.mean.shape == (n_rois, n_t)
        assert result.r_value.condition_a.mean.shape == (n_rois, n_t)
        assert result.roi_channel_counts.shape == (n_rois,)
        assert result.roi_subject_counts.shape == (n_rois,)
        # ROI_POS has 3 contributions (A1 from 01, A2 from 01, A1 from 02)
        roi_pos_idx = result.region_names.index("ROI_POS")
        assert result.roi_channel_counts[roi_pos_idx] == 3
        assert result.roi_subject_counts[roi_pos_idx] == 2
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_process_group_excludes_rois_below_thresholds() -> None:
    case_dir = _make_case_dir("exclude_rois")
    try:
        time_s = np.array([0.0, 0.1], dtype=np.float64)
        slope = np.ones((1, 2), dtype=np.float64)

        path_01 = case_dir / "sub-01_task-decid_desc-slopestat_stats.h5"
        _write_slope_stats_h5(path_01, channels=["A1"], time_s=time_s, condition_a_slope=slope, condition_b_slope=slope)
        file_01 = _make_bids_file(path_01, {"subject": "01", "task": "decid", "suffix": "stats", "extension": ".h5"})

        processor = RegressionGroupProcessing(
            RegressionGroupParams(
                roi_mode="manual",
                manual_region_channels={
                    "ROI_OK": {"01": ["A1"]},
                    "ROI_EMPTY": {"01": ["Z99"]},   # channel not in file
                    "ROI_NEEDS_2_SUBJ": {"01": ["A1"]},
                },
                min_subjects_per_roi=2,
            )
        )
        result = processor.process_group(BIDSFileGroup(primary=file_01))

        assert "ROI_EMPTY" in result.excluded_rois
        assert "ROI_NEEDS_2_SUBJ" in result.excluded_rois
        assert "ROI_OK" in result.excluded_rois  # also excluded: only 1 subject < 2
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_process_group_reports_missing_manual_channels(capsys: pytest.CaptureFixture[str]) -> None:
    case_dir = _make_case_dir("missing_manual_channels")
    try:
        time_s = np.array([0.0, 0.1], dtype=np.float64)
        slope = np.ones((1, 2), dtype=np.float64)

        path_01 = case_dir / "sub-01_task-decid_desc-slopestat_stats.h5"
        _write_slope_stats_h5(
            path_01,
            channels=["A1"],
            time_s=time_s,
            condition_a_slope=slope,
            condition_b_slope=slope,
        )
        file_01 = _make_bids_file(
            path_01,
            {"subject": "01", "task": "decid", "suffix": "stats", "extension": ".h5"},
        )

        processor = RegressionGroupProcessing(
            RegressionGroupParams(
                roi_mode="manual",
                manual_region_channels={
                    "ROI_WARN": {"01": ["A1", "Z99"], "02": ["B1"]},
                },
            )
        )
        result = processor.process_group(BIDSFileGroup(primary=file_01))
        captured = capsys.readouterr()

        assert result.manual_roi_missing_channels == {
            "ROI_WARN": {
                "01": ["Z99"],
                "02": ["B1"],
            }
        }
        assert (
            "Missing manual channels: ROI_WARN/01: Z99; ROI_WARN/02: B1"
            in captured.out
        )
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_process_group_output_entities_is_none() -> None:
    """Processor no longer sets output_entities; the writer injects subject='group' via _build_output_path."""
    case_dir = _make_case_dir("output_entities")
    try:
        time_s = np.array([0.0, 0.1], dtype=np.float64)
        slope = np.ones((1, 2), dtype=np.float64)
        path_01 = case_dir / "sub-01_task-decid_desc-slopestat_stats.h5"
        _write_slope_stats_h5(path_01, channels=["A1"], time_s=time_s, condition_a_slope=slope, condition_b_slope=slope)
        file_01 = _make_bids_file(path_01, {"subject": "01", "task": "decid", "suffix": "stats", "extension": ".h5"})

        processor = RegressionGroupProcessing(
            RegressionGroupParams(
                roi_mode="manual",
                manual_region_channels={"ROI_A": {"01": ["A1"]}},
            )
        )
        result = processor.process_group(BIDSFileGroup(primary=file_01))
        assert result.output_entities is None
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_process_group_contribution_samples_shape() -> None:
    case_dir = _make_case_dir("contrib_samples")
    try:
        time_s = np.array([0.0, 0.1, 0.2], dtype=np.float64)
        slope = np.array([[1.5, 1.5, 1.5], [2.0, 2.0, 2.0]], dtype=np.float64)
        path_01 = case_dir / "sub-01_task-decid_desc-slopestat_stats.h5"
        _write_slope_stats_h5(path_01, channels=["A1", "A2"], time_s=time_s, condition_a_slope=slope, condition_b_slope=-slope)
        file_01 = _make_bids_file(path_01, {"subject": "01", "task": "decid", "suffix": "stats", "extension": ".h5"})

        processor = RegressionGroupProcessing(
            RegressionGroupParams(
                roi_mode="manual",
                manual_region_channels={"ROI_A": {"01": ["A1", "A2"]}},
            )
        )
        result = processor.process_group(BIDSFileGroup(primary=file_01))

        assert len(result.slope_contributions.condition_a) == 1
        assert result.slope_contributions.condition_a[0].shape == (2, 3)
        assert len(result.slope_contributions.labels) == 1
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_process_group_cluster_permutation_custom_method() -> None:
    """cluster_permutation with method='custom' produces cluster p-values from permuted slopes."""
    case_dir = _make_case_dir("cluster_perm_custom")
    try:
        rng = np.random.default_rng(42)
        n_ch = 2
        n_t = 8
        n_perm = 20
        time_s = np.linspace(-0.2, 0.5, n_t)
        channels = ["A1", "A2"]

        files = []
        for i in range(3):
            path = case_dir / f"sub-0{i + 1}_task-decid_desc-slopestat_stats.h5"
            slope_a = rng.standard_normal((n_ch, n_t)).astype(np.float64)
            slope_b = rng.standard_normal((n_ch, n_t)).astype(np.float64)
            _write_slope_stats_h5(
                path, channels=channels, time_s=time_s,
                condition_a_slope=slope_a, condition_b_slope=slope_b,
            )
            perm_a = rng.standard_normal((n_perm, n_ch, n_t)).astype(np.float32)
            perm_b = rng.standard_normal((n_perm, n_ch, n_t)).astype(np.float32)
            with h5py.File(path, "a") as fh:
                fh["stats/regression/accepted"].create_dataset(
                    "permuted_slopes", data=perm_a, compression="gzip"
                )
                fh["stats/regression/rejected"].create_dataset(
                    "permuted_slopes", data=perm_b, compression="gzip"
                )
            files.append(_make_bids_file(path, {
                "subject": f"0{i + 1}", "task": "decid", "desc": "slopestat",
                "suffix": "stats", "extension": ".h5",
            }))

        processor = RegressionGroupProcessing(
            RegressionGroupParams(
                roi_mode="manual",
                manual_region_channels={"ROI1": {f"0{i + 1}": channels for i in range(3)}},
                p_value_correction_method="cluster_permutation",
                cluster_permutation_method="custom",
                n_group_permutations=50,
                permutation_seed=1,
            )
        )
        result = processor.process_group(BIDSFileGroup(primary=files[0], secondaries=files[1:]))

        assert result.cluster_p_values is not None
        assert result.cluster_p_values.shape == (1,)
        assert 0.0 <= float(result.cluster_p_values[0]) <= 1.0
        assert result.cluster_windows_s is not None
        assert len(result.cluster_windows_s) == 1
        assert isinstance(result.cluster_windows_s[0], list)
        assert result.cluster_null_distributions is not None
        assert len(result.cluster_null_distributions) == 1
        assert result.cluster_null_distributions[0].shape == (50,)
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_regression_group_params_raises_for_cluster_permutation_unpaired() -> None:
    """cluster_permutation + unpaired raises a ValidationError."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="cluster_permutation"):
        RegressionGroupParams(
            roi_mode="manual",
            manual_region_channels={"ROI": {"01": ["A1"]}},
            p_value_correction_method="cluster_permutation",
            contrast_mode="unpaired",
        )


def test_process_group_uses_primary_regression_metric_contract_for_r_value() -> None:
    case_dir = _make_case_dir("primary_regression_metric_r_value")
    try:
        time_s = np.array([0.0, 0.1, 0.2], dtype=np.float64)
        slope = np.array([[1.0, 1.0, 1.0]], dtype=np.float64)
        path_01 = case_dir / "sub-01_task-decid_desc-slopestat_stats.h5"
        _write_slope_stats_h5(
            path_01,
            channels=["A1"],
            time_s=time_s,
            condition_a_slope=slope,
            condition_b_slope=-slope,
            condition_a_r_value=np.array([[0.8, 0.7, 0.6]], dtype=np.float64),
            condition_b_r_value=np.array([[0.2, 0.1, 0.0]], dtype=np.float64),
        )
        path_02 = case_dir / "sub-02_task-decid_desc-slopestat_stats.h5"
        _write_slope_stats_h5(
            path_02,
            channels=["A1"],
            time_s=time_s,
            condition_a_slope=slope,
            condition_b_slope=-slope,
            condition_a_r_value=np.array([[0.9, 0.8, 0.7]], dtype=np.float64),
            condition_b_r_value=np.array([[0.1, 0.0, -0.1]], dtype=np.float64),
        )

        file_01 = _make_bids_file(path_01, {"subject": "01", "task": "decid", "suffix": "stats", "extension": ".h5"})
        file_02 = _make_bids_file(path_02, {"subject": "02", "task": "decid", "suffix": "stats", "extension": ".h5"})

        processor = RegressionGroupProcessing(
            RegressionGroupParams(
                roi_mode="manual",
                manual_region_channels={"ROI_A": {"01": ["A1"], "02": ["A1"]}},
                primary_regression_metric="r_value",
            )
        )
        result = processor.process_group(BIDSFileGroup(primary=file_01, secondaries=[file_02]))

        assert result.primary_regression_metric == "r_value"
        assert result.regression_stats.contrast.t_values.shape == (1, 3)
        assert result.slope.condition_a.mean.shape == (1, 3)
        np.testing.assert_allclose(
            result.r_value.condition_a.mean[0],
            np.array([0.85, 0.75, 0.65], dtype=np.float64),
        )
        np.testing.assert_allclose(
            result.r_value.condition_b.mean[0],
            np.array([0.15, 0.05, -0.05], dtype=np.float64),
        )
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)







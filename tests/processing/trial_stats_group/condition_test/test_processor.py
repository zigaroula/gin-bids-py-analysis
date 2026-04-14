from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import h5py
import numpy as np
import pytest
import scipy.io

from gin_bids_py_analysis.processing.utils.matlab import make_struct

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_stats_group import (
    ConditionTestGroupParams,
    ConditionTestGroupProcessing,
    build_condition_test_compatible_groups,
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


def _write_trial_stats_h5(
    path: Path,
    *,
    analysis_level: str = "channel",
    channels: list[str],
    time_s: np.ndarray,
    mean_difference: np.ndarray,
    t_values: np.ndarray | None = None,
    permuted_t_values: np.ndarray | None = None,
    condition_labels: tuple[str, str] = ("accepted", "rejected"),
    activity_zscore: str = "none",
    activity_baseline_tmin_s: float = -0.2,
    activity_baseline_tmax_s: float = 0.0,
    source_ieeg_files: list[str] | None = None,
    source_electrodes_files: list[str] | None = None,
) -> None:
    str_dtype = h5py.string_dtype(encoding="utf-8")
    p_values = np.full_like(mean_difference, 0.1, dtype=np.float64)
    t_values_arr = t_values if t_values is not None else np.full_like(mean_difference, 1.0, dtype=np.float64)

    with h5py.File(path, "w") as fh:
        stats_grp = fh.create_group("stats")
        stats_grp.create_dataset("t_values", data=t_values_arr)
        stats_grp.create_dataset("p_values", data=p_values)
        stats_grp.create_dataset("p_values_uncorrected", data=p_values)
        stats_grp.create_dataset("significant_mask", data=np.zeros_like(mean_difference, dtype=bool))
        if permuted_t_values is not None:
            stats_grp.create_dataset("permuted_t_values", data=permuted_t_values.astype(np.float32))

        means_grp = fh.create_group("means")
        means_grp.create_dataset(condition_labels[0], data=mean_difference + 1.0)
        means_grp.create_dataset(condition_labels[1], data=np.ones_like(mean_difference))
        means_grp.create_dataset("difference", data=mean_difference)

        uncertainty_grp = fh.create_group("uncertainty")
        uncertainty_grp.create_dataset(condition_labels[0] + "_sem", data=np.full_like(mean_difference, 0.2))
        uncertainty_grp.create_dataset(condition_labels[1] + "_sem", data=np.full_like(mean_difference, 0.2))
        uncertainty_grp.create_dataset("difference_sem", data=np.full_like(mean_difference, 0.3))
        uncertainty_grp.create_dataset("difference_ci95_low", data=mean_difference - 0.5)
        uncertainty_grp.create_dataset("difference_ci95_high", data=mean_difference + 0.5)

        axes_grp = fh.create_group("axes")
        axis_name = "region" if analysis_level == "roi" else "channel"
        axes_grp.create_dataset(axis_name, data=np.array(channels, dtype=object), dtype=str_dtype)
        axes_grp.create_dataset("time_s", data=time_s)

        meta_grp = fh.create_group("meta")
        meta_grp.create_dataset("analysis_level", data=analysis_level, dtype=str_dtype)
        meta_grp.create_dataset("trial_count_labels", data=np.array(condition_labels, dtype=object), dtype=str_dtype)
        meta_grp.create_dataset("trial_counts", data=np.array([12, 11], dtype=np.int64))
        meta_grp.create_dataset("binning_mode", data="none", dtype=str_dtype)
        meta_grp.create_dataset("window_ms", data=0.0)
        meta_grp.create_dataset("n_bins", data=0)
        meta_grp.create_dataset("effective_n_bins", data=int(len(time_s)))
        meta_grp.create_dataset("activity_zscore", data=activity_zscore, dtype=str_dtype)
        meta_grp.create_dataset("activity_baseline_tmin_s", data=activity_baseline_tmin_s)
        meta_grp.create_dataset("activity_baseline_tmax_s", data=activity_baseline_tmax_s)

        prov_grp = fh.create_group("provenance")
        prov_grp.create_dataset(
            "source_ieeg_files",
            data=np.array(source_ieeg_files or [], dtype=object),
            dtype=str_dtype,
        )
        prov_grp.create_dataset(
            "source_electrodes_files",
            data=np.array(source_electrodes_files or [], dtype=object),
            dtype=str_dtype,
        )


def test_build_condition_test_compatible_groups_splits_heterogeneous_inputs() -> None:
    case_dir = _make_case_dir("group_split")
    try:
        time_s = np.array([0.0, 0.1, 0.2], dtype=np.float64)
        data = np.ones((2, 3), dtype=np.float64)

        path_a = case_dir / "sub-01_task-decid_desc-conditiontest_stats.h5"
        path_b = case_dir / "sub-02_task-decid_desc-conditiontest_stats.h5"
        path_c = case_dir / "sub-03_task-other_desc-conditiontest_stats.h5"

        _write_trial_stats_h5(path_a, channels=["A1", "A2"], time_s=time_s, mean_difference=data)
        _write_trial_stats_h5(path_b, channels=["A1", "A2"], time_s=time_s, mean_difference=data)
        _write_trial_stats_h5(path_c, channels=["A1", "A2"], time_s=time_s, mean_difference=data)

        files = [
            _make_bids_file(path_a, {"subject": "01", "task": "decid", "desc": "conditiontest", "suffix": "stats", "extension": ".h5", "datatype": "ieeg"}),
            _make_bids_file(path_b, {"subject": "02", "task": "decid", "desc": "conditiontest", "suffix": "stats", "extension": ".h5", "datatype": "ieeg"}),
            _make_bids_file(path_c, {"subject": "03", "task": "other", "desc": "conditiontest", "suffix": "stats", "extension": ".h5", "datatype": "ieeg"}),
        ]

        groups = build_condition_test_compatible_groups(files, source_metric="mean_difference")
        assert len(groups) == 2
        assert sorted(len(group.all_files) for group in groups) == [1, 2]
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_build_condition_test_compatible_groups_splits_activity_zscore_inputs() -> None:
    case_dir = _make_case_dir("group_split_activity_zscore")
    try:
        time_s = np.array([0.0, 0.1, 0.2], dtype=np.float64)
        data = np.ones((2, 3), dtype=np.float64)

        path_a = case_dir / "sub-01_task-decid_desc-conditiontest_stats.h5"
        path_b = case_dir / "sub-02_task-decid_desc-conditiontest_stats.h5"
        _write_trial_stats_h5(path_a, channels=["A1", "A2"], time_s=time_s, mean_difference=data)
        _write_trial_stats_h5(
            path_b,
            channels=["A1", "A2"],
            time_s=time_s,
            mean_difference=data,
            activity_zscore="baseline",
        )

        files = [
            _make_bids_file(path_a, {"subject": "01", "task": "decid", "desc": "conditiontest", "suffix": "stats", "extension": ".h5", "datatype": "ieeg"}),
            _make_bids_file(path_b, {"subject": "02", "task": "decid", "desc": "conditiontest", "suffix": "stats", "extension": ".h5", "datatype": "ieeg"}),
        ]

        groups = build_condition_test_compatible_groups(files, source_metric="mean_difference")
        assert len(groups) == 2
        assert all(len(group.all_files) == 1 for group in groups)
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_process_group_rejects_non_channel_trial_stats() -> None:
    case_dir = _make_case_dir("reject_non_channel")
    try:
        stats_path = case_dir / "sub-01_task-decid_desc-conditiontest_stats.h5"
        _write_trial_stats_h5(
            stats_path,
            analysis_level="roi",
            channels=["ROI1"],
            time_s=np.array([0.0, 0.1], dtype=np.float64),
            mean_difference=np.ones((1, 2), dtype=np.float64),
        )

        file = _make_bids_file(
            stats_path,
            {"subject": "01", "task": "decid", "desc": "conditiontest", "suffix": "stats", "extension": ".h5", "datatype": "ieeg"},
        )
        processor = ConditionTestGroupProcessing(
            ConditionTestGroupParams(
                roi_mode="manual",
                manual_region_channels={"ROI1": {"01": ["ROI1"]}},
            )
        )

        with pytest.raises(ValueError, match="channel-level"):
            processor.process_group(BIDSFileGroup(primary=file))
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_process_group_manual_mode_and_thresholds() -> None:
    case_dir = _make_case_dir("manual_mode")
    try:
        time_s = np.array([0.0, 0.1, 0.2], dtype=np.float64)

        path_01 = case_dir / "sub-01_task-decid_desc-conditiontest_stats.h5"
        data_01 = np.array(
            [
                [2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0],
            ],
            dtype=np.float64,
        )
        _write_trial_stats_h5(path_01, channels=["A1", "A2", "B1"], time_s=time_s, mean_difference=data_01)

        path_02 = case_dir / "sub-02_task-decid_desc-conditiontest_stats.h5"
        data_02 = np.array(
            [
                [1.5, 1.5, 1.5],
                [-0.5, -0.5, -0.5],
            ],
            dtype=np.float64,
        )
        _write_trial_stats_h5(path_02, channels=["A1", "B1"], time_s=time_s, mean_difference=data_02)

        file_01 = _make_bids_file(path_01, {"subject": "01", "task": "decid", "desc": "conditiontest", "suffix": "stats", "extension": ".h5", "datatype": "ieeg"})
        file_02 = _make_bids_file(path_02, {"subject": "02", "task": "decid", "desc": "conditiontest", "suffix": "stats", "extension": ".h5", "datatype": "ieeg"})

        processor = ConditionTestGroupProcessing(
            ConditionTestGroupParams(
                roi_mode="manual",
                manual_region_channels={
                    "ROI_POS": {"01": ["A1", "A2"], "02": ["A1"]},
                    "ROI_NEG": {"01": ["B1"], "02": ["B1"]},
                    "ROI_DROP": {"01": ["A2"]},
                },
                min_subjects_per_roi=2,
            )
        )

        result = processor.process_group(BIDSFileGroup(primary=file_01, secondaries=[file_02]))

        assert result.region_names == ["ROI_POS", "ROI_NEG"]
        assert result.output_entities == {"subject": "group", "task": "decid"}
        assert result.activity_t_values.shape == (2, 3)
        assert np.all(result.metric_mean[0] > 0.0)
        assert np.all(result.metric_mean[1] < 0.0)
        assert "ROI_DROP" in result.excluded_rois
        assert result.condition_a_activity_mean.shape == (2, 3)
        assert result.condition_b_activity_mean.shape == (2, 3)
        assert result.condition_a_activity_sem.shape == (2, 3)
        assert result.condition_b_activity_sem.shape == (2, 3)
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_process_group_atlas_mode_uses_electrodes_mapping() -> None:
    case_dir = _make_case_dir("atlas_mode")
    try:
        ieeg_path = case_dir / "sub-01_task-decid_run-1_ieeg.vhdr"
        ieeg_path.write_text("", encoding="utf-8")

        electrodes_path = case_dir / "sub-01_task-decid_run-1_electrodes.tsv"
        electrodes_path.write_text(
            "name\tMarsAtlas\nA1\tROI_A\nB1\tROI_B\n",
            encoding="utf-8",
        )

        stats_path = case_dir / "sub-01_task-decid_desc-conditiontest_stats.h5"
        _write_trial_stats_h5(
            stats_path,
            channels=["A1", "B1"],
            time_s=np.array([0.0, 0.1], dtype=np.float64),
            mean_difference=np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64),
            source_ieeg_files=[str(ieeg_path)],
            source_electrodes_files=[str(electrodes_path)],
        )

        stats_file = _make_bids_file(
            stats_path,
            {
                "subject": "01",
                "task": "decid",
                "desc": "conditiontest",
                "suffix": "stats",
                "extension": ".h5",
                "datatype": "ieeg",
            },
        )

        processor = ConditionTestGroupProcessing(
            ConditionTestGroupParams(
                roi_mode="atlas",
                atlas_name="MarsAtlas",
            )
        )

        result = processor.process_group(BIDSFileGroup(primary=stats_file))

        assert result.region_names == ["ROI_A", "ROI_B"]
        assert list(result.roi_channel_counts) == [1, 1]
        assert result.source_electrodes_files == [str(electrodes_path)]
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_process_group_atlas_mode_with_nonexistent_ieeg_path() -> None:
    """Atlas mode works when source_ieeg_files paths no longer exist on disk.

    In visualization flows the iEEG files are preloaded into memory and the
    group processor only ever sees them as string paths in provenance.
    Entity extraction for the electrode lookup is pure filename parsing and
    must not require the physical file to be present.
    """
    case_dir = _make_case_dir("atlas_mode_no_ieeg_on_disk")
    try:
        # The iEEG path is intentionally NOT created on disk.
        ieeg_path = case_dir / "sub-01_task-decid_run-1_ieeg.vhdr"

        electrodes_path = case_dir / "sub-01_task-decid_run-1_electrodes.tsv"
        electrodes_path.write_text(
            "name\tMarsAtlas\nA1\tROI_A\nB1\tROI_B\n",
            encoding="utf-8",
        )

        stats_path = case_dir / "sub-01_task-decid_desc-conditiontest_stats.h5"
        _write_trial_stats_h5(
            stats_path,
            channels=["A1", "B1"],
            time_s=np.array([0.0, 0.1], dtype=np.float64),
            mean_difference=np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64),
            source_ieeg_files=[str(ieeg_path)],
            source_electrodes_files=[str(electrodes_path)],
        )

        stats_file = _make_bids_file(
            stats_path,
            {
                "subject": "01",
                "task": "decid",
                "desc": "conditiontest",
                "suffix": "stats",
                "extension": ".h5",
                "datatype": "ieeg",
            },
        )

        assert not ieeg_path.exists(), "precondition: iEEG file must not exist"

        processor = ConditionTestGroupProcessing(
            ConditionTestGroupParams(
                roi_mode="atlas",
                atlas_name="MarsAtlas",
            )
        )

        result = processor.process_group(BIDSFileGroup(primary=stats_file))

        assert result.region_names == ["ROI_A", "ROI_B"]
        assert list(result.roi_channel_counts) == [1, 1]
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_process_group_cluster_permutation_custom_mode() -> None:
    case_dir = _make_case_dir("cluster_custom")
    try:
        time_s = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
        perm_rng = np.random.default_rng(10)
        perm_01 = perm_rng.normal(0.0, 1.0, size=(20, 1, len(time_s))).astype(np.float64)
        perm_02 = perm_rng.normal(0.0, 1.0, size=(20, 1, len(time_s))).astype(np.float64)

        path_01 = case_dir / "sub-01_task-decid_desc-conditiontest_stats.h5"
        _write_trial_stats_h5(
            path_01,
            channels=["A1"],
            time_s=time_s,
            mean_difference=np.array([[2.0, 2.0, 2.0, 0.5]], dtype=np.float64),
            permuted_t_values=perm_01,
        )
        path_02 = case_dir / "sub-02_task-decid_desc-conditiontest_stats.h5"
        _write_trial_stats_h5(
            path_02,
            channels=["A1"],
            time_s=time_s,
            mean_difference=np.array([[2.5, 2.5, 2.5, 0.5]], dtype=np.float64),
            permuted_t_values=perm_02,
        )

        file_01 = _make_bids_file(
            path_01,
            {"subject": "01", "task": "decid", "desc": "conditiontest", "suffix": "stats", "extension": ".h5", "datatype": "ieeg"},
        )
        file_02 = _make_bids_file(
            path_02,
            {"subject": "02", "task": "decid", "desc": "conditiontest", "suffix": "stats", "extension": ".h5", "datatype": "ieeg"},
        )

        processor = ConditionTestGroupProcessing(
            ConditionTestGroupParams(
                roi_mode="manual",
                manual_region_channels={"ROI_A": {"01": ["A1"], "02": ["A1"]}},
                p_value_correction_method="cluster_permutation",
                cluster_permutation_method="custom",
                n_group_permutations=40,
                permutation_seed=123,
            )
        )
        result = processor.process_group(BIDSFileGroup(primary=file_01, secondaries=[file_02]))

        assert result.cluster_p_values is not None
        assert result.cluster_p_values.shape == (1,)
        assert result.cluster_best_cluster_windows_s is not None
        assert len(result.cluster_best_cluster_windows_s) == 1
        assert result.cluster_null_distributions is not None
        assert result.cluster_null_distributions[0].shape == (40,)
        assert result.activity_p_values.shape == (1, len(time_s))
        assert result.activity_significant_mask.shape == (1, len(time_s))
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_process_group_cluster_permutation_mne_mode() -> None:
    case_dir = _make_case_dir("cluster_mne")
    try:
        time_s = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64)
        path_01 = case_dir / "sub-01_task-decid_desc-conditiontest_stats.h5"
        _write_trial_stats_h5(
            path_01,
            channels=["A1"],
            time_s=time_s,
            mean_difference=np.array([[2.0, 2.0, 2.0, 0.0]], dtype=np.float64),
        )
        path_02 = case_dir / "sub-02_task-decid_desc-conditiontest_stats.h5"
        _write_trial_stats_h5(
            path_02,
            channels=["A1"],
            time_s=time_s,
            mean_difference=np.array([[2.1, 2.1, 2.1, 0.0]], dtype=np.float64),
        )

        file_01 = _make_bids_file(
            path_01,
            {"subject": "01", "task": "decid", "desc": "conditiontest", "suffix": "stats", "extension": ".h5", "datatype": "ieeg"},
        )
        file_02 = _make_bids_file(
            path_02,
            {"subject": "02", "task": "decid", "desc": "conditiontest", "suffix": "stats", "extension": ".h5", "datatype": "ieeg"},
        )

        processor = ConditionTestGroupProcessing(
            ConditionTestGroupParams(
                roi_mode="manual",
                manual_region_channels={"ROI_A": {"01": ["A1"], "02": ["A1"]}},
                p_value_correction_method="cluster_permutation",
                cluster_permutation_method="mne",
                n_group_permutations=40,
                permutation_seed=321,
            )
        )
        result = processor.process_group(BIDSFileGroup(primary=file_01, secondaries=[file_02]))

        assert result.cluster_p_values is not None
        assert result.cluster_p_values.shape == (1,)
        assert result.cluster_best_cluster_windows_s is not None
        assert len(result.cluster_best_cluster_windows_s) == 1
        assert result.cluster_null_distributions is not None
        assert result.cluster_null_distributions[0].ndim == 1
        assert result.activity_p_values.shape == (1, len(time_s))
        assert result.activity_significant_mask.shape == (1, len(time_s))
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Helpers and tests for .mat input files
# ---------------------------------------------------------------------------

def _write_trial_stats_mat(
    path: Path,
    *,
    analysis_level: str = "channel",
    channels: list[str],
    time_s: np.ndarray,
    mean_difference: np.ndarray,
    t_values: np.ndarray | None = None,
    condition_labels: tuple[str, str] = ("accepted", "rejected"),
    source_ieeg_files: list[str] | None = None,
    source_electrodes_files: list[str] | None = None,
) -> None:
    """Write a minimal trial-stats .mat fixture matching the TrialStatsProcessingWriter schema."""
    t_arr = t_values if t_values is not None else np.full_like(mean_difference, 1.0, dtype=np.float64)
    p_arr = np.full_like(mean_difference, 0.1, dtype=np.float64)
    sig_arr = np.zeros_like(mean_difference, dtype=np.uint8)

    stats_struct = make_struct(
        t_values=t_arr.astype(np.float64),
        p_values=p_arr.astype(np.float64),
        p_values_uncorrected=p_arr.astype(np.float64),
        significant_mask=sig_arr,
    )
    means_struct = make_struct(
        **{
            condition_labels[0]: (mean_difference + 1.0).astype(np.float64),
            condition_labels[1]: np.ones_like(mean_difference, dtype=np.float64),
            "difference": mean_difference.astype(np.float64),
        }
    )

    primary_axis_name = "region" if analysis_level == "roi" else "channel"
    axes_struct = make_struct(
        **{
            primary_axis_name: np.array(channels, dtype=object),
            "time_s": time_s.astype(np.float64),
        }
    )
    meta_struct = make_struct(
        trial_counts=np.array([12, 11], dtype=np.int64),
        trial_count_labels=np.array(list(condition_labels), dtype=object),
        sampling_frequency_hz=512.0,
        p_value_correction_method=np.str_("none"),
        significance_alpha=0.05,
        analysis_level=np.str_(analysis_level),
        atlas_name=np.str_(""),
        atlas_regions=np.array([], dtype=object),
        window_ms=0.0,
        n_bins=0,
        effective_n_bins=int(len(time_s)),
        binning_mode=np.str_("none"),
        activity_zscore=np.str_("none"),
        activity_baseline_tmin_s=-0.2,
        activity_baseline_tmax_s=0.0,
        stats_valid=np.uint8(1),
    )
    prov_struct = make_struct(
        source_ieeg_files=np.array(source_ieeg_files or [], dtype=object),
        source_electrodes_files=np.array(source_electrodes_files or [], dtype=object),
        pipeline_name=np.str_("conditiontest"),
        pipeline_version=np.str_("test"),
    )
    data = make_struct(
        stats=stats_struct,
        means=means_struct,
        axes=axes_struct,
        meta=meta_struct,
        provenance=prov_struct,
    )
    scipy.io.savemat(str(path), {"data": data}, do_compression=True)


def test_process_group_manual_mode_mat_input() -> None:
    """Group processor should produce the same result from .mat as from .h5 fixtures."""
    case_dir = _make_case_dir("manual_mode_mat")
    try:
        time_s = np.array([0.0, 0.1, 0.2], dtype=np.float64)

        path_01 = case_dir / "sub-01_task-decid_desc-conditiontest_stats.mat"
        data_01 = np.array(
            [
                [2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0],
            ],
            dtype=np.float64,
        )
        _write_trial_stats_mat(path_01, channels=["A1", "A2", "B1"], time_s=time_s, mean_difference=data_01)

        path_02 = case_dir / "sub-02_task-decid_desc-conditiontest_stats.mat"
        data_02 = np.array(
            [
                [1.5, 1.5, 1.5],
                [-0.5, -0.5, -0.5],
            ],
            dtype=np.float64,
        )
        _write_trial_stats_mat(path_02, channels=["A1", "B1"], time_s=time_s, mean_difference=data_02)

        file_01 = _make_bids_file(
            path_01,
            {"subject": "01", "task": "decid", "desc": "conditiontest", "suffix": "stats", "extension": ".mat", "datatype": "ieeg"},
        )
        file_02 = _make_bids_file(
            path_02,
            {"subject": "02", "task": "decid", "desc": "conditiontest", "suffix": "stats", "extension": ".mat", "datatype": "ieeg"},
        )

        processor = ConditionTestGroupProcessing(
            ConditionTestGroupParams(
                roi_mode="manual",
                manual_region_channels={
                    "ROI_POS": {"01": ["A1", "A2"], "02": ["A1"]},
                    "ROI_NEG": {"01": ["B1"], "02": ["B1"]},
                    "ROI_DROP": {"01": ["A2"]},
                },
                min_subjects_per_roi=2,
            )
        )

        result = processor.process_group(BIDSFileGroup(primary=file_01, secondaries=[file_02]))

        assert result.region_names == ["ROI_POS", "ROI_NEG"]
        assert result.output_entities == {"subject": "group", "task": "decid"}
        assert result.activity_t_values.shape == (2, 3)
        assert np.all(result.metric_mean[0] > 0.0)
        assert np.all(result.metric_mean[1] < 0.0)
        assert "ROI_DROP" in result.excluded_rois
        assert result.condition_a_activity_mean.shape == (2, 3)
        assert result.condition_b_activity_mean.shape == (2, 3)
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_build_compatible_groups_mat_files() -> None:
    """build_condition_test_compatible_groups should split heterogeneous .mat inputs correctly."""
    case_dir = _make_case_dir("group_split_mat")
    try:
        time_s = np.array([0.0, 0.1, 0.2], dtype=np.float64)
        data = np.ones((2, 3), dtype=np.float64)

        path_a = case_dir / "sub-01_task-decid_desc-conditiontest_stats.mat"
        path_b = case_dir / "sub-02_task-decid_desc-conditiontest_stats.mat"
        path_c = case_dir / "sub-03_task-other_desc-conditiontest_stats.mat"

        _write_trial_stats_mat(path_a, channels=["A1", "A2"], time_s=time_s, mean_difference=data)
        _write_trial_stats_mat(path_b, channels=["A1", "A2"], time_s=time_s, mean_difference=data)
        _write_trial_stats_mat(path_c, channels=["A1", "A2"], time_s=time_s, mean_difference=data)

        files = [
            _make_bids_file(path_a, {"subject": "01", "task": "decid", "desc": "conditiontest", "suffix": "stats", "extension": ".mat", "datatype": "ieeg"}),
            _make_bids_file(path_b, {"subject": "02", "task": "decid", "desc": "conditiontest", "suffix": "stats", "extension": ".mat", "datatype": "ieeg"}),
            _make_bids_file(path_c, {"subject": "03", "task": "other", "desc": "conditiontest", "suffix": "stats", "extension": ".mat", "datatype": "ieeg"}),
        ]

        groups = build_condition_test_compatible_groups(files, source_metric="mean_difference")
        assert len(groups) == 2
        assert sorted(len(group.all_files) for group in groups) == [1, 2]
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)

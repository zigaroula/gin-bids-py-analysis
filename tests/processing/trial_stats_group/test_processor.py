from __future__ import annotations

import shutil
import uuid
from pathlib import Path

import h5py
import numpy as np
import pytest

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_stats_group import (
    TrialStatsGroupParams,
    TrialStatsGroupProcessing,
    build_trial_stats_compatible_groups,
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
    condition_labels: tuple[str, str] = ("accepted", "rejected"),
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


def test_build_trial_stats_compatible_groups_splits_heterogeneous_inputs() -> None:
    case_dir = _make_case_dir("group_split")
    try:
        time_s = np.array([0.0, 0.1, 0.2], dtype=np.float64)
        data = np.ones((2, 3), dtype=np.float64)

        path_a = case_dir / "sub-01_task-decid_desc-trialstats_stats.h5"
        path_b = case_dir / "sub-02_task-decid_desc-trialstats_stats.h5"
        path_c = case_dir / "sub-03_task-other_desc-trialstats_stats.h5"

        _write_trial_stats_h5(path_a, channels=["A1", "A2"], time_s=time_s, mean_difference=data)
        _write_trial_stats_h5(path_b, channels=["A1", "A2"], time_s=time_s, mean_difference=data)
        _write_trial_stats_h5(path_c, channels=["A1", "A2"], time_s=time_s, mean_difference=data)

        files = [
            _make_bids_file(path_a, {"subject": "01", "task": "decid", "desc": "trialstats", "suffix": "stats", "extension": ".h5", "datatype": "ieeg"}),
            _make_bids_file(path_b, {"subject": "02", "task": "decid", "desc": "trialstats", "suffix": "stats", "extension": ".h5", "datatype": "ieeg"}),
            _make_bids_file(path_c, {"subject": "03", "task": "other", "desc": "trialstats", "suffix": "stats", "extension": ".h5", "datatype": "ieeg"}),
        ]

        groups = build_trial_stats_compatible_groups(files, source_metric="mean_difference")
        assert len(groups) == 2
        assert sorted(len(group.all_files) for group in groups) == [1, 2]
    finally:
        shutil.rmtree(case_dir, ignore_errors=True)


def test_process_group_rejects_non_channel_trial_stats() -> None:
    case_dir = _make_case_dir("reject_non_channel")
    try:
        stats_path = case_dir / "sub-01_task-decid_desc-trialstats_stats.h5"
        _write_trial_stats_h5(
            stats_path,
            analysis_level="roi",
            channels=["ROI1"],
            time_s=np.array([0.0, 0.1], dtype=np.float64),
            mean_difference=np.ones((1, 2), dtype=np.float64),
        )

        file = _make_bids_file(
            stats_path,
            {"subject": "01", "task": "decid", "desc": "trialstats", "suffix": "stats", "extension": ".h5", "datatype": "ieeg"},
        )
        processor = TrialStatsGroupProcessing(
            TrialStatsGroupParams(
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

        path_01 = case_dir / "sub-01_task-decid_desc-trialstats_stats.h5"
        data_01 = np.array(
            [
                [2.0, 2.0, 2.0],
                [1.0, 1.0, 1.0],
                [-1.0, -1.0, -1.0],
            ],
            dtype=np.float64,
        )
        _write_trial_stats_h5(path_01, channels=["A1", "A2", "B1"], time_s=time_s, mean_difference=data_01)

        path_02 = case_dir / "sub-02_task-decid_desc-trialstats_stats.h5"
        data_02 = np.array(
            [
                [1.5, 1.5, 1.5],
                [-0.5, -0.5, -0.5],
            ],
            dtype=np.float64,
        )
        _write_trial_stats_h5(path_02, channels=["A1", "B1"], time_s=time_s, mean_difference=data_02)

        file_01 = _make_bids_file(path_01, {"subject": "01", "task": "decid", "desc": "trialstats", "suffix": "stats", "extension": ".h5", "datatype": "ieeg"})
        file_02 = _make_bids_file(path_02, {"subject": "02", "task": "decid", "desc": "trialstats", "suffix": "stats", "extension": ".h5", "datatype": "ieeg"})

        processor = TrialStatsGroupProcessing(
            TrialStatsGroupParams(
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
        assert result.t_values.shape == (2, 3)
        assert np.all(result.metric_mean[0] > 0.0)
        assert np.all(result.metric_mean[1] < 0.0)
        assert "ROI_DROP" in result.excluded_rois
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

        stats_path = case_dir / "sub-01_task-decid_desc-trialstats_stats.h5"
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
                "desc": "trialstats",
                "suffix": "stats",
                "extension": ".h5",
                "datatype": "ieeg",
            },
        )

        processor = TrialStatsGroupProcessing(
            TrialStatsGroupParams(
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

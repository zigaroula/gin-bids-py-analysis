from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.processing.time_frequency_stats.condition_test.result import (
    TFConditionContrast,
    TFDifferenceEstimate,
    TimeFrequencyConditionTestResult,
)
from bidsforge.processing.time_frequency_stats.regression.result import (
    TFConditionPredictorValues,
    TFConditionRegressionStats,
    TFPredictorValues,
    TFRegressionStats,
    TimeFrequencyRegressionResult,
)
from bidsforge.processing.time_frequency_stats.result import TFConditionEstimate, TFConditionPair
from bidsforge.processing.time_frequency_stats_group.condition_test import (
    TimeFrequencyConditionTestGroupParams,
    TimeFrequencyConditionTestGroupProcessing,
    TimeFrequencyConditionTestGroupWriter,
    TimeFrequencyConditionTestGroupWriterParams,
    build_time_frequency_condition_test_compatible_groups,
    load_time_frequency_condition_test_group_result,
)
from bidsforge.processing.time_frequency_stats_group.regression import (
    TimeFrequencyRegressionGroupParams,
    TimeFrequencyRegressionGroupProcessing,
    TimeFrequencyRegressionGroupWriter,
    TimeFrequencyRegressionGroupWriterParams,
    build_time_frequency_regression_compatible_groups,
    load_time_frequency_regression_group_result,
)
from bidsforge.processing.utils.serialization import write_hdf5_tree
from bidsforge.processing.utils.time_frequency_group_stats import (
    label_tf_clusters,
    matlab_style_cluster_mask,
)


FREQUENCY_HZ = np.array([5.0, 10.0], dtype=np.float64)
TIME_S = np.array([0.0, 0.1, 0.2], dtype=np.float64)


def test_condition_test_group_manual_roi_and_writer_roundtrip(tmp_path: Path) -> None:
    file_01 = _write_condition_subject(
        tmp_path,
        subject="01",
        channels=["A1", "A2"],
        t_values=np.stack([_map(2.0), _map(4.0)], axis=0),
    )
    file_02 = _write_condition_subject(
        tmp_path,
        subject="02",
        channels=["A1"],
        t_values=np.stack([_map(8.0)], axis=0),
    )
    params = TimeFrequencyConditionTestGroupParams(
        roi_mode="manual",
        manual_region_channels={"ROI_A": {"01": ["A1", "A2"], "02": ["A1"]}},
        primary_condition_metric="t_values",
        p_value_correction_method="none",
    )

    result = TimeFrequencyConditionTestGroupProcessing(params).process_group(
        BIDSFileGroup(primary=file_01, secondaries=[file_02])
    )

    assert result.region_names == ["ROI_A"]
    assert result.roi_channel_counts.tolist() == [3]
    assert result.roi_subject_counts.tolist() == [2]
    np.testing.assert_allclose(result.source_metric.mean[0], _map(14.0 / 3.0))
    assert result.source_metric_contributions.labels == [["01/A1", "01/A2", "02/A1"]]

    writer = TimeFrequencyConditionTestGroupWriter(
        TimeFrequencyConditionTestGroupWriterParams(bids_root=tmp_path, output_format="hdf5")
    )
    output = writer.write(result)
    loaded = load_time_frequency_condition_test_group_result(output)
    np.testing.assert_allclose(loaded.source_metric.mean, result.source_metric.mean)
    assert loaded.region_names == ["ROI_A"]

    mat_writer = TimeFrequencyConditionTestGroupWriter(
        TimeFrequencyConditionTestGroupWriterParams(
            bids_root=tmp_path,
            output_format="matlab",
            output_description="tfconditiontestgroupmat",
        )
    )
    mat_output = mat_writer.write(result)
    loaded_mat = load_time_frequency_condition_test_group_result(mat_output)
    np.testing.assert_allclose(loaded_mat.source_metric.mean, result.source_metric.mean)


def test_regression_group_manual_roi_contrast_and_writer_roundtrip(tmp_path: Path) -> None:
    file_01 = _write_regression_subject(
        tmp_path,
        subject="01",
        channels=["A1", "A2"],
        metric_a=np.stack([_map(4.0), _map(8.0)], axis=0),
        metric_b=np.stack([_map(1.0), _map(2.0)], axis=0),
    )
    file_02 = _write_regression_subject(
        tmp_path,
        subject="02",
        channels=["A1"],
        metric_a=np.stack([_map(10.0)], axis=0),
        metric_b=np.stack([_map(4.0)], axis=0),
    )
    params = TimeFrequencyRegressionGroupParams(
        roi_mode="manual",
        manual_region_channels={"ROI_A": {"01": ["A1", "A2"], "02": ["A1"]}},
        primary_regression_metric="t_values",
        p_value_correction_method="none",
    )

    result = TimeFrequencyRegressionGroupProcessing(params).process_group(
        BIDSFileGroup(primary=file_01, secondaries=[file_02])
    )

    np.testing.assert_allclose(result.source_metric.condition_a.mean[0], _map(22.0 / 3.0))
    np.testing.assert_allclose(result.source_metric.condition_b.mean[0], _map(7.0 / 3.0))
    assert result.stats.condition_contrast.t_values.shape == (1, 2, 3)
    assert result.source_metric_contributions.condition_a[0].shape == (3, 2, 3)

    writer = TimeFrequencyRegressionGroupWriter(
        TimeFrequencyRegressionGroupWriterParams(bids_root=tmp_path, output_format="hdf5")
    )
    output = writer.write(result)
    loaded = load_time_frequency_regression_group_result(output)
    np.testing.assert_allclose(
        loaded.source_metric.condition_a.mean,
        result.source_metric.condition_a.mean,
    )

    mat_writer = TimeFrequencyRegressionGroupWriter(
        TimeFrequencyRegressionGroupWriterParams(
            bids_root=tmp_path,
            output_format="matlab",
            output_description="tfregressiongroupmat",
        )
    )
    mat_output = mat_writer.write(result)
    loaded_mat = load_time_frequency_regression_group_result(mat_output)
    np.testing.assert_allclose(
        loaded_mat.source_metric.condition_a.mean,
        result.source_metric.condition_a.mean,
    )


def test_compatible_group_helpers_split_axes_and_predictor(tmp_path: Path) -> None:
    cond_a = _write_condition_subject(tmp_path, subject="01", channels=["A1"], t_values=np.ones((1, 2, 3)))
    cond_b = _write_condition_subject(
        tmp_path,
        subject="02",
        channels=["A1"],
        t_values=np.ones((1, 2, 2)),
        time_s=np.array([0.0, 0.1], dtype=np.float64),
    )
    assert len(build_time_frequency_condition_test_compatible_groups([cond_a, cond_b])) == 2

    reg_a = _write_regression_subject(
        tmp_path,
        subject="03",
        channels=["A1"],
        metric_a=np.ones((1, 2, 3)),
        metric_b=np.ones((1, 2, 3)),
        predictor="gain",
    )
    reg_b = _write_regression_subject(
        tmp_path,
        subject="04",
        channels=["A1"],
        metric_a=np.ones((1, 2, 3)),
        metric_b=np.ones((1, 2, 3)),
        predictor="loss",
    )
    assert len(build_time_frequency_regression_compatible_groups([reg_a, reg_b])) == 2


def test_atlas_mode_uses_electrodes_mapping(tmp_path: Path) -> None:
    electrodes = tmp_path / "sub-01" / "ieeg" / "sub-01_electrodes.tsv"
    electrodes.parent.mkdir(parents=True, exist_ok=True)
    electrodes.write_text("name\tMarsAtlas\nA1\tROI Left\nA2\tn/a\n", encoding="utf-8")
    file_01 = _write_condition_subject(
        tmp_path,
        subject="01",
        channels=["A1", "A2"],
        t_values=np.stack([_map(5.0), _map(99.0)], axis=0),
        source_electrodes_files=[str(electrodes)],
    )
    params = TimeFrequencyConditionTestGroupParams(
        roi_mode="atlas",
        atlas_name="MarsAtlas",
        p_value_correction_method="none",
    )

    result = TimeFrequencyConditionTestGroupProcessing(params).process_group(
        BIDSFileGroup(primary=file_01)
    )

    assert result.region_names == ["ROI_Left"]
    np.testing.assert_allclose(result.source_metric.mean[0], _map(5.0))
    assert result.source_electrodes_files == [str(electrodes)]


def test_cluster_helpers_use_8_connectivity_and_source_map_sum() -> None:
    mask = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool)
    labels, n_labels = label_tf_clusters(mask)
    assert n_labels == 1
    assert np.all(labels[mask] == 1)

    source = np.array([[2.0, 0.0], [0.0, 3.0]])
    p_values = np.array([[0.01, 0.9], [0.9, 0.01]])
    sig, cluster_labels, cluster_sums = matlab_style_cluster_mask(
        source_map=source,
        p_values=p_values,
        null_distribution=np.array([-4.0, -3.0, 3.0, 4.0]),
        cluster_threshold_alpha=0.05,
        cluster_percentile_alpha=0.25,
    )
    assert cluster_labels.max() == 1
    np.testing.assert_allclose(cluster_sums, np.array([5.0]))
    assert bool(np.all(sig[p_values <= 0.05]))


def test_custom_cluster_requires_subject_permutations(tmp_path: Path) -> None:
    file_01 = _write_condition_subject(
        tmp_path,
        subject="01",
        channels=["A1"],
        t_values=np.stack([_map(1.0)], axis=0),
    )
    params = TimeFrequencyConditionTestGroupParams(
        roi_mode="manual",
        manual_region_channels={"ROI_A": {"01": ["A1"]}},
        p_value_correction_method="cluster_permutation",
        cluster_permutation_method="custom",
    )
    with pytest.raises(ValueError, match="requires permuted_t_values"):
        TimeFrequencyConditionTestGroupProcessing(params).process_group(
            BIDSFileGroup(primary=file_01)
        )


def _map(value: float, *, freqs: int = 2, times: int = 3) -> np.ndarray:
    return np.full((freqs, times), value, dtype=np.float64)


def _write_condition_subject(
    tmp_path: Path,
    *,
    subject: str,
    channels: list[str],
    t_values: np.ndarray,
    frequency_hz: np.ndarray = FREQUENCY_HZ,
    time_s: np.ndarray = TIME_S,
    source_electrodes_files: list[str] | None = None,
) -> BIDSFile:
    path = (
        tmp_path
        / "derivatives"
        / "time_frequency_condition_test"
        / f"sub-{subject}"
        / "ieeg"
        / f"sub-{subject}_task-test_desc-tfconditiontest_stats.h5"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = tmp_path / f"sub-{subject}" / "ieeg" / f"sub-{subject}_task-test_ieeg.vhdr"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("", encoding="utf-8")
    signal_a = t_values + 10.0
    signal_b = np.full_like(t_values, 10.0)
    result = TimeFrequencyConditionTestResult(
        source_group=BIDSFileGroup(primary=BIDSFile.from_path(raw)),
        metadata={
            "analysis_type": "time_frequency_condition_test",
            "time_selection": "strict_matlab",
            "baseline_grand_average": False,
        },
        channel_names=channels,
        frequency_hz=frequency_hz,
        time_axis_s=time_s,
        condition_a="accepted",
        condition_b="rejected",
        condition_a_trial_count=12,
        condition_b_trial_count=11,
        source_tfr_file=str(raw),
        source_ieeg_files=[str(raw)],
        source_electrodes_files=source_electrodes_files or [],
        signal_activity=TFConditionPair(
            condition_a=TFConditionEstimate(mean=signal_a, sem=np.ones_like(signal_a)),
            condition_b=TFConditionEstimate(mean=signal_b, sem=np.ones_like(signal_b)),
        ),
        stats_valid=True,
        power_mode="stored",
        p_value_correction_method="none",
        significance_alpha=0.05,
        difference=TFDifferenceEstimate(mean=signal_a - signal_b, sem=np.ones_like(signal_a)),
        contrast=TFConditionContrast(
            t_values=t_values,
            p_values=np.full_like(t_values, 0.1),
            p_values_uncorrected=np.full_like(t_values, 0.1),
            significant_mask=np.zeros_like(t_values, dtype=bool),
        ),
    )
    write_hdf5_tree(path, result.to_output_tree())
    return BIDSFile.from_path(path)


def _write_regression_subject(
    tmp_path: Path,
    *,
    subject: str,
    channels: list[str],
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    frequency_hz: np.ndarray = FREQUENCY_HZ,
    time_s: np.ndarray = TIME_S,
    predictor: str = "gain",
) -> BIDSFile:
    path = (
        tmp_path
        / "derivatives"
        / "time_frequency_regression"
        / f"sub-{subject}"
        / "ieeg"
        / f"sub-{subject}_task-test_desc-tfregression_stats.h5"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = tmp_path / f"sub-{subject}" / "ieeg" / f"sub-{subject}_task-test_ieeg.vhdr"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text("", encoding="utf-8")
    signal_a = metric_a + 20.0
    signal_b = metric_b + 20.0
    result = TimeFrequencyRegressionResult(
        source_group=BIDSFileGroup(primary=BIDSFile.from_path(raw)),
        metadata={
            "analysis_type": "time_frequency_regression",
            "time_selection": "strict_matlab",
            "baseline_grand_average": False,
        },
        channel_names=channels,
        frequency_hz=frequency_hz,
        time_axis_s=time_s,
        condition_a="accepted",
        condition_b="rejected",
        condition_a_trial_count=12,
        condition_b_trial_count=11,
        source_tfr_file=str(raw),
        source_ieeg_files=[str(raw)],
        signal_activity=TFConditionPair(
            condition_a=TFConditionEstimate(mean=signal_a, sem=np.ones_like(signal_a)),
            condition_b=TFConditionEstimate(mean=signal_b, sem=np.ones_like(signal_b)),
        ),
        stats_valid=True,
        power_mode="stored",
        p_value_correction_method="none",
        significance_alpha=0.05,
        regression=TFRegressionStats(
            condition_a=_regression_stats(metric_a),
            condition_b=_regression_stats(metric_b),
        ),
        predictor_values=TFPredictorValues(
            condition_a=TFConditionPredictorValues(
                raw_values=np.array([1.0, 2.0]),
                transformed_values=np.array([1.0, 2.0]),
                values=np.array([1.0, 2.0]),
            ),
            condition_b=TFConditionPredictorValues(
                raw_values=np.array([1.0, 2.0]),
                transformed_values=np.array([1.0, 2.0]),
                values=np.array([1.0, 2.0]),
            ),
        ),
        predictor=predictor,
        predictor_zscore="none",
    )
    write_hdf5_tree(path, result.to_output_tree())
    return BIDSFile.from_path(path)


def _regression_stats(metric: np.ndarray) -> TFConditionRegressionStats:
    return TFConditionRegressionStats(
        slope=metric / 10.0,
        intercept=np.zeros_like(metric),
        r_value=metric / 100.0,
        t_values=metric,
        p_value=np.full_like(metric, 0.1),
        p_value_corrected=np.full_like(metric, 0.1),
        significant_mask=np.zeros_like(metric, dtype=bool),
        n_trials_used=12,
        stats_valid=True,
    )

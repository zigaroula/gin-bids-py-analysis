from __future__ import annotations

from pathlib import Path

import numpy as np

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.processing.time_frequency.result import TimeFrequencyProcessingResult
from bidsforge.processing.time_frequency_stats.regression import (
    TimeFrequencyRegressionParams,
    TimeFrequencyRegressionProcessing,
    TimeFrequencyRegressionWriter,
    TimeFrequencyRegressionWriterParams,
    load_time_frequency_regression_result,
)
from bidsforge.processing.trial_stats import TableTrialResolver
from bidsforge.processing.utils.serialization import write_hdf5_tree


def test_regression_fits_tfr_slope_and_writes_roundtrip(tmp_path: Path) -> None:
    predictor = np.array([0.0, 1.0, 2.0, 0.0, 1.0, 2.0], dtype=np.float32)
    labels = ["A", "A", "A", "B", "B", "B"]
    slope_a = np.full((1, 2, 3), 2.0, dtype=np.float32)
    slope_b = np.full((1, 2, 3), -1.5, dtype=np.float32)
    intercept_a = np.full((1, 2, 3), 1.0, dtype=np.float32)
    intercept_b = np.full((1, 2, 3), 4.0, dtype=np.float32)
    power = np.empty((6, 1, 2, 3), dtype=np.float32)
    for idx, value in enumerate(predictor):
        if labels[idx] == "A":
            power[idx] = intercept_a + value * slope_a
        else:
            power[idx] = intercept_b + value * slope_b

    group = _make_group(tmp_path, power, labels, predictor)
    params = TimeFrequencyRegressionParams(
        condition_a="condition_a",
        condition_b="condition_b",
        predictor="rating_z",
        predictor_zscore="none",
        p_value_correction_method="none",
    )
    result = TimeFrequencyRegressionProcessing(params, resolver=_resolver()).process_group(group)

    np.testing.assert_allclose(result.regression.condition_a.slope, slope_a, atol=1e-12)
    np.testing.assert_allclose(result.regression.condition_b.slope, slope_b, atol=1e-12)
    assert result.regression.condition_a.stats_valid
    assert result.regression.condition_b.stats_valid
    np.testing.assert_allclose(result.predictor_values.condition_a.values, [0.0, 1.0, 2.0])

    writer = TimeFrequencyRegressionWriter(
        TimeFrequencyRegressionWriterParams(
            bids_root=tmp_path,
            output_format="hdf5",
            output_description="tfregression",
        )
    )
    output_path = writer.write(result)
    loaded = load_time_frequency_regression_result(output_path)
    np.testing.assert_allclose(loaded.regression.condition_a.slope, result.regression.condition_a.slope)

    mat_writer = TimeFrequencyRegressionWriter(
        TimeFrequencyRegressionWriterParams(
            bids_root=tmp_path,
            output_format="matlab",
            output_description="tfregressionmat",
        )
    )
    mat_output_path = mat_writer.write(result)
    loaded_mat = load_time_frequency_regression_result(mat_output_path)
    np.testing.assert_allclose(loaded_mat.regression.condition_a.slope, result.regression.condition_a.slope)


def _resolver() -> TableTrialResolver:
    return TableTrialResolver(
        conditions=[
            {"label": "condition_a", "when": {"column": "condition", "op": "==", "value": "A"}},
            {"label": "condition_b", "when": {"column": "condition", "op": "==", "value": "B"}},
        ],
        extract_columns=["condition", "rating_z"],
        filter={"suffix": "beh"},
    )


def _make_group(
    tmp_path: Path,
    power: np.ndarray,
    labels: list[str],
    predictor: np.ndarray,
) -> BIDSFileGroup:
    tfr_path = tmp_path / "derivatives" / "time_frequency" / "sub-01" / "ieeg" / "sub-01_task-test_desc-tf_tfr.h5"
    tfr_path.parent.mkdir(parents=True)
    raw_path = tmp_path / "sub-01" / "ieeg" / "sub-01_task-test_ieeg.vhdr"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text("", encoding="utf-8")
    result = TimeFrequencyProcessingResult(
        source_group=BIDSFileGroup(primary=BIDSFile.from_path(raw_path)),
        power_db=power,
        baseline_db=np.zeros(power.shape[:3], dtype=np.float32),
        trial_ids=[str(i) for i in range(power.shape[0])],
        channel_names=["C1"],
        frequency_hz=np.array([5.0, 10.0]),
        time_s=np.array([0.0, 1.0, 2.0]),
        original_fs=100.0,
        metadata={"apply_baseline": False},
    )
    write_hdf5_tree(tfr_path, result.to_output_tree())

    beh_path = tmp_path / "sub-01" / "beh" / "sub-01_beh.tsv"
    beh_path.parent.mkdir(parents=True)
    lines = ["condition\trating_z"]
    lines.extend(f"{label}\t{float(value)}" for label, value in zip(labels, predictor))
    beh_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return BIDSFileGroup(
        primary=BIDSFile.from_path(tfr_path),
        secondaries=[BIDSFile.from_path(beh_path)],
    )

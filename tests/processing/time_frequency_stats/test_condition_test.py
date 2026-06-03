from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.processing.time_frequency.result import TimeFrequencyProcessingResult
from bidsforge.processing.time_frequency_stats.condition_test import (
    TimeFrequencyConditionTestParams,
    TimeFrequencyConditionTestProcessing,
    TimeFrequencyConditionTestWriter,
    TimeFrequencyConditionTestWriterParams,
    load_time_frequency_condition_test_result,
)
from bidsforge.processing.trial_stats import TableTrialResolver
from bidsforge.processing.utils.serialization import write_hdf5_tree


def test_condition_test_splits_tfr_trials_and_writes_roundtrip(tmp_path: Path) -> None:
    power = np.arange(6 * 2 * 3 * 4, dtype=np.float32).reshape(6, 2, 3, 4)
    power[3:] += 10.0
    group = _make_group(tmp_path, power, ["A", "A", "A", "B", "B", "B"])

    params = TimeFrequencyConditionTestParams(
        time_window_s=(0.0, 3.0),
        time_selection="strict_matlab",
        condition_a="condition_a",
        condition_b="condition_b",
        p_value_correction_method="none",
        compute_grand_average=True,
    )
    processor = TimeFrequencyConditionTestProcessing(params, resolver=_resolver())
    result = processor.process_group(group)

    assert result.contrast.t_values.shape == (2, 3, 2)
    expected = ttest_ind(
        power[:3, :, :, 1:3],
        power[3:, :, :, 1:3],
        axis=0,
        equal_var=False,
    ).statistic
    np.testing.assert_allclose(result.contrast.t_values, expected)
    np.testing.assert_allclose(result.signal_activity.condition_a.mean, np.mean(power[:3, :, :, 1:3], axis=0))
    assert result.grand_average is not None
    np.testing.assert_allclose(result.grand_average.mean, np.mean(power[:, :, :, 1:3], axis=0))

    writer = TimeFrequencyConditionTestWriter(
        TimeFrequencyConditionTestWriterParams(
            bids_root=tmp_path,
            output_format="hdf5",
            output_description="tfconditiontest",
        )
    )
    output_path = writer.write(result)
    loaded = load_time_frequency_condition_test_result(output_path)
    np.testing.assert_allclose(loaded.contrast.t_values, result.contrast.t_values)
    assert loaded.channel_names == ["C1", "C2"]

    mat_writer = TimeFrequencyConditionTestWriter(
        TimeFrequencyConditionTestWriterParams(
            bids_root=tmp_path,
            output_format="matlab",
            output_description="tfconditiontestmat",
        )
    )
    mat_output_path = mat_writer.write(result)
    loaded_mat = load_time_frequency_condition_test_result(mat_output_path)
    np.testing.assert_allclose(loaded_mat.contrast.t_values, result.contrast.t_values)


def _resolver() -> TableTrialResolver:
    return TableTrialResolver(
        conditions=[
            {"label": "condition_a", "when": {"column": "condition", "op": "==", "value": "A"}},
            {"label": "condition_b", "when": {"column": "condition", "op": "==", "value": "B"}},
        ],
        extract_columns=["condition"],
        filter={"suffix": "beh"},
    )


def _make_group(tmp_path: Path, power: np.ndarray, labels: list[str]) -> BIDSFileGroup:
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
        channel_names=["C1", "C2"],
        frequency_hz=np.array([5.0, 10.0, 20.0]),
        time_s=np.array([0.0, 1.0, 2.0, 3.0]),
        original_fs=100.0,
        metadata={"apply_baseline": False},
    )
    write_hdf5_tree(tfr_path, result.to_output_tree())

    beh_path = tmp_path / "sub-01" / "beh" / "sub-01_beh.tsv"
    beh_path.parent.mkdir(parents=True)
    rows = ["condition"] + labels
    beh_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return BIDSFileGroup(
        primary=BIDSFile.from_path(tfr_path),
        secondaries=[BIDSFile.from_path(beh_path)],
    )

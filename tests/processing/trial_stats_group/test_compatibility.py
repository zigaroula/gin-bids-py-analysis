from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.processing.trial_stats import ConditionTestProcessingResult
from bidsforge.processing.trial_stats_group.compatibility import (
    build_subject_stats_signature,
)


class _MockPyBIDSFile:
    def __init__(self, path: str, entities: dict[str, str]) -> None:
        self.path = path
        self.entities = entities


def _make_bids_file(path: Path, entities: dict[str, str]) -> BIDSFile:
    return BIDSFile(_MockPyBIDSFile(str(path), entities))


def _make_result() -> ConditionTestProcessingResult:
    primary = _make_bids_file(
        Path("sub-01_task-decid_ieeg.vhdr"),
        {"subject": "01", "task": "decid"},
    )
    return ConditionTestProcessingResult(
        source_group=BIDSFileGroup(primary=primary),
        time_axis_s=np.array([-0.2, 0.0, 0.2], dtype=np.float64),
        channel_names=["A1", "A2"],
        condition_a="accepted",
        condition_b="rejected",
        analysis_level="channel",
        window_ms=100.0,
        n_bins=0,
        activity_zscore="baseline",
        activity_baseline_tmin_s=-0.2,
        activity_baseline_tmax_s=0.0,
        activity_baseline_scope="global",
        activity_baseline_remove_outlier_trial_means=True,
        trial_activity_summary_kind="anchor_to_response_mean",
        trial_activity_summary_missing_response_policy="clamp_to_epoch",
        trial_activity_summary_source={"column": "rt", "source": "table_column"},
        trial_activity_summary_label="Mean activity",
        metadata={"binning_mode": "window_ms", "effective_n_bins": 3},
    )


def _signature(result: ConditionTestProcessingResult, **extra: object):
    file = _make_bids_file(
        Path("sub-01_task-decid_desc-conditiontest_stats.h5"),
        {"subject": "01", "task": "decid", "desc": "conditiontest"},
    )
    return build_subject_stats_signature(file, result, extra_key_parts=extra or None)


def test_identical_results_have_identical_signature() -> None:
    result_a = _make_result()
    result_b = deepcopy(result_a)

    assert _signature(result_a).key == _signature(result_b).key


def test_common_semantic_fields_change_signature() -> None:
    base = _make_result()
    variants = []

    changed_time = deepcopy(base)
    changed_time.time_axis_s = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
    variants.append(changed_time)

    changed_level = deepcopy(base)
    changed_level.analysis_level = "roi"
    variants.append(changed_level)

    changed_labels = deepcopy(base)
    changed_labels.condition_a = "pleasant"
    variants.append(changed_labels)

    changed_baseline = deepcopy(base)
    changed_baseline.activity_baseline_tmin_s = -0.5
    variants.append(changed_baseline)

    changed_summary = deepcopy(base)
    changed_summary.trial_activity_summary_kind = "epoch_mean"
    variants.append(changed_summary)

    base_key = _signature(base).key
    assert all(_signature(variant).key != base_key for variant in variants)


def test_extra_key_parts_change_signature() -> None:
    result = _make_result()

    assert _signature(result, primary_condition_metric="t_values").key != _signature(
        result,
        primary_condition_metric="mean_difference",
    ).key


def test_extra_key_parts_are_serialized_stably() -> None:
    result = _make_result()
    a = _signature(
        result,
        predictor_transform_by_condition={
            "accepted": {"offset": 1.0, "scale": 2.0},
            "rejected": {"scale": 3.0, "offset": 4.0},
        },
    )
    b = _signature(
        result,
        predictor_transform_by_condition={
            "rejected": {"offset": 4.0, "scale": 3.0},
            "accepted": {"scale": 2.0, "offset": 1.0},
        },
    )

    assert a.extra_key_parts == b.extra_key_parts
    assert a.key == b.key

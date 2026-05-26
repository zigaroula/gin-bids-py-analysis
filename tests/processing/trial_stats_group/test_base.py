from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.processing.trial_stats_group.processor import (
    BaseTrialStatsGroupContributionRecord,
    BaseTrialStatsGroupSnapshot,
    BaseTrialStatsGroupSnapshotSignature,
    build_compatible_groups,
    collect_manual_roi_records,
    find_missing_manual_roi_channels,
    format_manual_roi_missing_channels_message,
    hash_time_axis,
    validate_group_compatibility,
)


class _MockPyBIDSFile:
    def __init__(self, path: str, entities: dict[str, str]) -> None:
        self.path = path
        self.entities = entities


def _make_bids_file(path: Path, entities: dict[str, str]) -> BIDSFile:
    return BIDSFile(_MockPyBIDSFile(str(path), entities))


def _make_signature(*, task: str = "decid", source_desc: str = "conditiontest") -> BaseTrialStatsGroupSnapshotSignature:
    time_axis = np.array([-0.2, 0.0, 0.2], dtype=np.float64)
    return BaseTrialStatsGroupSnapshotSignature(
        task=task,
        source_desc=source_desc,
        condition_labels=("accepted", "rejected"),
        time_axis_hash=hash_time_axis(time_axis),
        time_axis_len=len(time_axis),
        binning_mode="none",
        window_ms=0.0,
        n_bins=0,
        effective_n_bins=len(time_axis),
        analysis_level="channel",
    )


def _make_snapshot(
    *,
    path: str,
    subject: str = "01",
    analysis_level: str = "channel",
) -> BaseTrialStatsGroupSnapshot:
    signature = _make_signature()
    return BaseTrialStatsGroupSnapshot(
        stats_file=_make_bids_file(
            Path(path),
            {
                "subject": subject,
                "task": "decid",
                "desc": "conditiontest",
                "suffix": "stats",
                "extension": ".h5",
            },
        ),
        subject=subject,
        task="decid",
        source_desc="conditiontest",
        condition_labels=("accepted", "rejected"),
        channel_names=["A1", "A2"],
        channel_index_by_norm={"a1": 0, "a2": 1},
        time_axis_s=np.array([-0.2, 0.0, 0.2], dtype=np.float64),
        analysis_level=analysis_level,
        binning_mode="none",
        window_ms=0.0,
        n_bins=0,
        effective_n_bins=3,
        source_ieeg_files=[],
        source_electrodes_files=[],
        signature=signature,
    )


def test_build_compatible_groups_splits_by_signature_key(tmp_path: Path) -> None:
    file_a = _make_bids_file(tmp_path / "sub-01_desc-a_stats.h5", {"subject": "01"})
    file_b = _make_bids_file(tmp_path / "sub-02_desc-a_stats.h5", {"subject": "02"})
    file_c = _make_bids_file(tmp_path / "sub-03_desc-b_stats.h5", {"subject": "03"})

    signature_a = _make_signature(source_desc="conditiontest")
    signature_b = _make_signature(source_desc="regression")

    groups = build_compatible_groups(
        [file_c, file_b, file_a],
        read_signature=lambda file: (
            signature_b if "desc-b" in str(file.path) else signature_a
        ),
    )

    assert len(groups) == 2
    assert [item.path.name for item in groups[0].all_files] == [
        "sub-01_desc-a_stats.h5",
        "sub-02_desc-a_stats.h5",
    ]
    assert [item.path.name for item in groups[1].all_files] == ["sub-03_desc-b_stats.h5"]


def test_validate_group_compatibility_rejects_non_channel_inputs() -> None:
    snapshots = [
        _make_snapshot(path="sub-01_stats.h5", subject="01"),
        _make_snapshot(path="sub-02_stats.h5", subject="02", analysis_level="roi"),
    ]

    with pytest.raises(ValueError, match="non-channel"):
        validate_group_compatibility(
            snapshots,
            empty_message="empty",
            non_channel_message="non-channel",
            incompatible_message="incompatible",
        )


def test_collect_manual_roi_records_normalizes_subject_and_channel_matching() -> None:
    snapshot = _make_snapshot(path="sub-01_stats.h5", subject="sub-01")

    records = collect_manual_roi_records(
        snapshots=[snapshot],
        manual_region_channels={"Insula": {"01": ["a1", "A2", "missing"]}},
        create_record=lambda roi, subject, snap, idx: BaseTrialStatsGroupContributionRecord(
            roi=roi,
            subject=subject,
            channel=snap.channel_names[idx],
            source_stats_file=str(snap.stats_file.path),
        ),
    )

    assert [record.channel for record in records["Insula"]] == ["A1", "A2"]
    assert all(record.subject == "01" for record in records["Insula"])


def test_find_missing_manual_roi_channels_reports_subject_and_channel_misses() -> None:
    snapshot = _make_snapshot(path="sub-01_stats.h5", subject="sub-01")

    missing = find_missing_manual_roi_channels(
        snapshots=[snapshot],
        manual_region_channels={
            "Insula": {"01": ["A1", "A3"], "02": ["B1"]},
        },
    )

    assert missing == {
        "Insula": {
            "01": ["A3"],
            "02": ["B1"],
        }
    }


def test_format_manual_roi_missing_channels_message_is_compact() -> None:
    message = format_manual_roi_missing_channels_message(
        {
            "Insula": {
                "01": ["A3"],
                "02": ["B1", "B2"],
            }
        }
    )

    assert message == (
        "Missing manual channels: "
        "Insula/01: A3; Insula/02: B1, B2"
    )


def test_hash_time_axis_is_stable_for_identical_values() -> None:
    axis_a = np.array([-0.1, 0.0, 0.1], dtype=np.float64)
    axis_b = np.array([-0.1, 0.0, 0.1], dtype=np.float64)

    assert hash_time_axis(axis_a) == hash_time_axis(axis_b)





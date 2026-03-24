from __future__ import annotations

import pytest

from gin_bids_py_analysis.processing.trial_stats_group import TrialStatsGroupParams


def test_params_require_atlas_name_in_atlas_mode() -> None:
    with pytest.raises(ValueError, match="atlas_name is required"):
        TrialStatsGroupParams(roi_mode="atlas")


def test_params_require_manual_mapping_in_manual_mode() -> None:
    with pytest.raises(ValueError, match="manual_region_channels is required"):
        TrialStatsGroupParams(roi_mode="manual")


def test_params_forbid_atlas_name_in_manual_mode() -> None:
    with pytest.raises(ValueError, match="atlas_name cannot be set"):
        TrialStatsGroupParams(
            roi_mode="manual",
            atlas_name="MarsAtlas",
            manual_region_channels={"ROI": {"01": ["A1"]}},
        )


def test_params_coerce_manual_mapping_and_subject_labels() -> None:
    params = TrialStatsGroupParams(
        roi_mode="manual",
        manual_region_channels={
            " ROI1 ": {"sub-01": ["A1", "A1", " A2 "]},
            "ROI2": {"02": "B1"},
        },
    )

    assert params.manual_region_channels == {
        "ROI1": {"01": ["A1", "A2"]},
        "ROI2": {"02": ["B1"]},
    }

from __future__ import annotations

import pytest

from gin_bids_py_analysis.processing.trial_stats_group import ConditionTestGroupParams


def test_params_require_atlas_name_in_atlas_mode() -> None:
    with pytest.raises(ValueError, match="atlas_name is required"):
        ConditionTestGroupParams(roi_mode="atlas")


def test_params_require_manual_mapping_in_manual_mode() -> None:
    with pytest.raises(ValueError, match="manual_region_channels is required"):
        ConditionTestGroupParams(roi_mode="manual")


def test_params_forbid_atlas_name_in_manual_mode() -> None:
    with pytest.raises(ValueError, match="atlas_name cannot be set"):
        ConditionTestGroupParams(
            roi_mode="manual",
            atlas_name="MarsAtlas",
            manual_region_channels={"ROI": {"01": ["A1"]}},
        )


def test_params_coerce_manual_mapping_and_subject_labels() -> None:
    params = ConditionTestGroupParams(
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


def test_params_accept_new_cluster_method_names() -> None:
    params = ConditionTestGroupParams(
        roi_mode="manual",
        manual_region_channels={"ROI": {"01": ["A1"]}},
        cluster_permutation_method="sign_flip",
    )
    assert params.cluster_permutation_method == "sign_flip"


@pytest.mark.parametrize("legacy_name", ["hierarchical", "mne"])
def test_params_reject_legacy_cluster_method_names(legacy_name: str) -> None:
    with pytest.raises(ValueError, match="custom' or 'sign_flip"):
        ConditionTestGroupParams(
            roi_mode="manual",
            manual_region_channels={"ROI": {"01": ["A1"]}},
            cluster_permutation_method=legacy_name,  # type: ignore[arg-type]
        )


def test_params_roi_name_forbidden_chars_are_replaced() -> None:
    params = ConditionTestGroupParams(
        roi_mode="manual",
        manual_region_channels={"my roi/left": {"01": ["A1"]}},
    )
    assert "my_roi_left" in params.manual_region_channels


def test_params_roi_name_starting_with_digit_is_rejected() -> None:
    with pytest.raises(ValueError, match="starts with a digit"):
        ConditionTestGroupParams(
            roi_mode="manual",
            manual_region_channels={"1_bad_roi": {"01": ["A1"]}},
        )


def test_params_roi_name_exceeding_63_chars_is_rejected() -> None:
    long_name = "a" * 64
    with pytest.raises(ValueError, match="63"):
        ConditionTestGroupParams(
            roi_mode="manual",
            manual_region_channels={long_name: {"01": ["A1"]}},
        )


def test_params_roi_names_collision_after_sanitization_is_rejected() -> None:
    with pytest.raises(ValueError, match="already used by another ROI"):
        ConditionTestGroupParams(
            roi_mode="manual",
            manual_region_channels={
                "my roi": {"01": ["A1"]},
                "my-roi": {"02": ["B1"]},
            },
        )



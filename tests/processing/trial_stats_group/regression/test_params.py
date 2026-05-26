from __future__ import annotations

import pytest

from gin_bids_py_analysis.processing.trial_stats_group import (
    RegressionGroupParams,
    RegressionGroupWriterParams,
)


def test_params_defaults() -> None:
    params = RegressionGroupParams(
        roi_mode="manual",
        manual_region_channels={"ROI_A": {"01": ["A1"]}},
    )
    assert params.p_value_correction_method == "none"
    assert params.significance_alpha == 0.05
    assert params.min_channels_per_roi == 1
    assert params.min_subjects_per_roi == 1


def test_params_atlas_mode_requires_atlas_name() -> None:
    with pytest.raises(ValueError, match="atlas_name"):
        RegressionGroupParams(roi_mode="atlas")


def test_params_manual_mode_requires_region_channels() -> None:
    with pytest.raises(ValueError, match="manual_region_channels"):
        RegressionGroupParams(roi_mode="manual")


def test_params_coerces_manual_region_channels_keys() -> None:
    params = RegressionGroupParams(
        roi_mode="manual",
        manual_region_channels={"ROI_A": {"sub-01": ["A1", "A2"]}},
    )
    # sub- prefix should be stripped from subject keys
    assert "01" in params.manual_region_channels["ROI_A"]


def test_params_atlas_mode_accepted() -> None:
    params = RegressionGroupParams(
        roi_mode="atlas",
        atlas_name="Destrieux",
    )
    assert params.atlas_name == "Destrieux"
    assert params.roi_mode == "atlas"


def test_params_correction_methods_accepted() -> None:
    for method in ("none", "fdr_bh", "bonferroni"):
        params = RegressionGroupParams(
            roi_mode="manual",
            manual_region_channels={"ROI_A": {"01": ["A1"]}},
            p_value_correction_method=method,
        )
        assert params.p_value_correction_method == method


def test_writer_params_defaults() -> None:
    from pathlib import Path

    wp = RegressionGroupWriterParams(bids_root=Path("/tmp"))
    assert wp.pipeline_label == "regression_group"
    assert wp.output_description == "regressiongroup"
    assert wp.output_format == "hdf5"
    assert wp.output_extension == ".h5"


def test_writer_params_matlab_extension() -> None:
    from pathlib import Path

    wp = RegressionGroupWriterParams(bids_root=Path("/tmp"), output_format="matlab")
    assert wp.output_extension == ".mat"


def test_params_roi_name_forbidden_chars_are_replaced() -> None:
    params = RegressionGroupParams(
        roi_mode="manual",
        manual_region_channels={"my roi/left": {"01": ["A1"]}},
    )
    assert "my_roi_left" in params.manual_region_channels


def test_params_roi_name_starting_with_digit_is_rejected() -> None:
    with pytest.raises(ValueError, match="starts with a digit"):
        RegressionGroupParams(
            roi_mode="manual",
            manual_region_channels={"1_bad_roi": {"01": ["A1"]}},
        )


def test_params_roi_name_exceeding_63_chars_is_rejected() -> None:
    long_name = "a" * 64
    with pytest.raises(ValueError, match="63"):
        RegressionGroupParams(
            roi_mode="manual",
            manual_region_channels={long_name: {"01": ["A1"]}},
        )


def test_params_roi_names_collision_after_sanitization_is_rejected() -> None:
    with pytest.raises(ValueError, match="already used by another ROI"):
        RegressionGroupParams(
            roi_mode="manual",
            manual_region_channels={
                "my roi": {"01": ["A1"]},
                "my-roi": {"02": ["B1"]},
            },
        )



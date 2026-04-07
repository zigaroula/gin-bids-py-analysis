from __future__ import annotations

import pytest

from gin_bids_py_analysis.processing.trial_slope_stats_group import (
    TrialSlopeStatsGroupParams,
    TrialSlopeStatsGroupWriterParams,
)


def test_params_defaults() -> None:
    params = TrialSlopeStatsGroupParams(
        roi_mode="manual",
        manual_region_channels={"ROI_A": {"01": ["A1"]}},
    )
    assert params.p_value_correction_method == "none"
    assert params.significance_alpha == 0.05
    assert params.min_channels_per_roi == 1
    assert params.min_subjects_per_roi == 1


def test_params_atlas_mode_requires_atlas_name() -> None:
    with pytest.raises(ValueError, match="atlas_name"):
        TrialSlopeStatsGroupParams(roi_mode="atlas")


def test_params_manual_mode_requires_region_channels() -> None:
    with pytest.raises(ValueError, match="manual_region_channels"):
        TrialSlopeStatsGroupParams(roi_mode="manual")


def test_params_coerces_manual_region_channels_keys() -> None:
    params = TrialSlopeStatsGroupParams(
        roi_mode="manual",
        manual_region_channels={"ROI_A": {"sub-01": ["A1", "A2"]}},
    )
    # sub- prefix should be stripped from subject keys
    assert "01" in params.manual_region_channels["ROI_A"]


def test_params_atlas_mode_accepted() -> None:
    params = TrialSlopeStatsGroupParams(
        roi_mode="atlas",
        atlas_name="Destrieux",
    )
    assert params.atlas_name == "Destrieux"
    assert params.roi_mode == "atlas"


def test_params_correction_methods_accepted() -> None:
    for method in ("none", "fdr_bh", "bonferroni"):
        params = TrialSlopeStatsGroupParams(
            roi_mode="manual",
            manual_region_channels={"ROI_A": {"01": ["A1"]}},
            p_value_correction_method=method,
        )
        assert params.p_value_correction_method == method


def test_writer_params_defaults() -> None:
    from pathlib import Path

    wp = TrialSlopeStatsGroupWriterParams(bids_root=Path("/tmp"))
    assert wp.pipeline_label == "trial_slope_stats_group"
    assert wp.output_description == "trialslopestatsgroup"
    assert wp.output_format == "hdf5"
    assert wp.output_extension == ".h5"


def test_writer_params_matlab_extension() -> None:
    from pathlib import Path

    wp = TrialSlopeStatsGroupWriterParams(bids_root=Path("/tmp"), output_format="matlab")
    assert wp.output_extension == ".mat"

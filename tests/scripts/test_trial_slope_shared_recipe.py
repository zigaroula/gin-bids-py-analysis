from __future__ import annotations

from dataclasses import replace

from scripts import trial_slope_shared as shared
from gin_bids_py_analysis.processing.utils.trial_annotator import (
    TRIAL_FEATURE_NAN_MASKS_KEY,
    EventAnnotationFeatureMaskRule,
)
from gin_bids_py_analysis.processing.utils.trial_resolver import ResolvedTrial


def test_recipe_derives_pipeline_descriptions() -> None:
    assert shared.RECIPE.hilbert_derivative_description == (
        f"{shared.HILBERT_OUTPUT_DESCRIPTION}"
        f"sm{shared.HILBERT_SMOOTHING_WINDOW_MS_FOR_STATS}"
    )
    assert shared.IEEG_FILTERS["desc"] == shared.RECIPE.hilbert_derivative_description
    assert (
        shared.TRIAL_SLOPE_STATS_FILTERS["desc"]
        == shared.RECIPE.trial_slope_output_description
    )
    assert (
        shared.GROUP_STATS_FILTERS["desc"]
        == shared.RECIPE.trial_slope_output_description
    )


def test_recipe_can_disable_hilbert_notch_filter() -> None:
    recipe = replace(shared.RECIPE, enable_hilbert_notch_filter=False)

    assert recipe.build_hilbert_params().notch_filter_freqs == []


def test_preset_overrides_can_switch_cleanly() -> None:
    settings = [
        "HILBERT_NOTCH_FILTER_FREQS",
        "HILBERT_OUTPUT_DESCRIPTION",
        "TRIAL_SLOPE_OUTPUT_DESCRIPTION",
        "USE_DELPHOS_SPIKE_FILTER",
    ]
    original_values = {name: getattr(shared, name) for name in settings}

    try:
        assert shared._apply_trial_slope_preset("50Hz") == "50hz"
        assert shared.HILBERT_NOTCH_FILTER_FREQS == [50.0]
        assert shared.HILBERT_OUTPUT_DESCRIPTION == "bga50hz"
        assert shared.TRIAL_SLOPE_OUTPUT_DESCRIPTION == "onset50hz"
        assert shared.USE_DELPHOS_SPIKE_FILTER is False

        assert shared._apply_trial_slope_preset("delphos") == "delphos"
        assert shared.HILBERT_NOTCH_FILTER_FREQS == []
        assert shared.HILBERT_OUTPUT_DESCRIPTION == "bga"
        assert shared.TRIAL_SLOPE_OUTPUT_DESCRIPTION == "onsetdelphos"
        assert shared.USE_DELPHOS_SPIKE_FILTER is True

        assert shared._apply_trial_slope_preset("regular") == "regular"
        assert shared.HILBERT_NOTCH_FILTER_FREQS == []
        assert shared.HILBERT_OUTPUT_DESCRIPTION == "bga"
        assert shared.TRIAL_SLOPE_OUTPUT_DESCRIPTION == "onset"
        assert shared.USE_DELPHOS_SPIKE_FILTER is False
    finally:
        for name, value in original_values.items():
            setattr(shared, name, value)


def test_combine_manual_region_channels_merges_sources_and_deduplicates() -> None:
    manual_region_channels = {
        "vaINS": {"01": ["A1", "A2"], "02": ["B1"]},
        "daINS": {"01": ["A2", "C1"], "03": ["D1"]},
        "vmPFC": {"01": ["P1"]},
    }

    combined = shared.combine_manual_region_channels(
        manual_region_channels,
        {"aIns": ["vaINS", "daINS"]},
    )

    assert combined == {
        "vmPFC": {"01": ["P1"]},
        "aIns": {
            "01": ["A1", "A2", "C1"],
            "02": ["B1"],
            "03": ["D1"],
        },
    }


def test_recipe_build_group_params_applies_configured_roi_combinations() -> None:
    recipe = replace(
        shared.RECIPE,
        group_roi_combinations={"aIns": ["vaINS", "daINS"]},
        group_keep_combined_source_rois=False,
        group_param_kwargs={
            **shared.GROUP_PARAM_KWARGS,
            "p_value_correction_method": "none",
        },
    )
    manual_region_channels = {
        "vaINS": {"01": ["A1"]},
        "daINS": {"02": ["B1"]},
        "vmPFC": {"01": ["P1"]},
    }

    params = recipe.build_regression_group_params(manual_region_channels)

    assert params.manual_region_channels == {
        "vmPFC": {"01": ["P1"]},
        "aIns": {"01": ["A1"], "02": ["B1"]},
    }


def test_default_group_config_keeps_independent_insula_rois() -> None:
    assert shared.ROI_CSV_FILES["aIns"].name == "aINS_b5_finite_channels.csv"
    assert shared.ROI_CSV_FILES["daINS"].name == "aINS_dors_elecs_tbl.csv"
    assert shared.ROI_CSV_FILES["vaINS"].name == "aINS_vent_elecs_tbl.csv"
    assert shared.GROUP_ROI_COMBINATIONS == {}


def test_recipe_builds_delphos_filter_from_selected_rois() -> None:
    recipe = replace(
        shared.RECIPE,
        use_delphos_spike_filter=True,
        use_matlab_zscores=False,
        delphos_spike_filter_rois=["vmPFC", "daINS"],
    )
    manual_region_channels = {
        "vmPFC": {"01": ["A1"]},
        "daINS": {"01": ["B1"], "02": ["C1"]},
        "vaINS": {"01": ["D1"]},
    }

    annotators = recipe.build_trial_annotators(manual_region_channels)
    invalidation_rule = annotators[-1]
    event_filter = invalidation_rule.event_filter

    assert invalidation_rule.exclusion_reason == shared.VM_PFC_SPIKE_EXCLUSION_REASON
    assert event_filter.any is not None
    rendered = str(event_filter.model_dump())
    assert "A1" in rendered
    assert "B1" in rendered
    assert "C1" in rendered
    assert "D1" not in rendered


def test_recipe_can_mask_only_delphos_channels() -> None:
    recipe = replace(
        shared.RECIPE,
        use_delphos_spike_filter=True,
        use_matlab_zscores=False,
        delphos_spike_filter_rois=["vmPFC"],
        delphos_spike_filter_mode="channel",
    )
    manual_region_channels = {"vmPFC": {"01": ["A1"]}}

    annotators = recipe.build_trial_annotators(manual_region_channels)
    feature_mask_rule = annotators[-1]
    trial = ResolvedTrial(
        source_file=None,  # type: ignore[arg-type]
        anchor_event_index=0,
        anchor_event_code="11",
        anchor_onset_s=0.0,
        anchor_duration_s=0.0,
        metadata={
            "delphos_events": [
                {"subject": "01", "event_type": "Spike", "channel": "A1"},
                {"subject": "01", "event_type": "Spike", "channel": "B1"},
            ],
        },
    )

    assert isinstance(feature_mask_rule, EventAnnotationFeatureMaskRule)
    feature_mask_rule.annotate_trials(
        group=None,  # type: ignore[arg-type]
        ieeg_file=None,  # type: ignore[arg-type]
        trials=[trial],
        tmin_s=-0.5,
        tmax_s=5.0,
        ieeg_channel_names=["A1", "B1"],
    )

    assert trial.keep is True
    assert trial.metadata[TRIAL_FEATURE_NAN_MASKS_KEY] == [
        {"features": ["A1"], "reason": shared.VM_PFC_SPIKE_EXCLUSION_REASON}
    ]

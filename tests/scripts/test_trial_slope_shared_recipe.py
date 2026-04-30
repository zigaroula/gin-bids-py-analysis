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

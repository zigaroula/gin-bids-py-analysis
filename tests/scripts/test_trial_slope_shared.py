from __future__ import annotations

from pathlib import Path

import sys


SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import trial_slope_shared as shared  # noqa: E402


def _make_recipe(output_format: str) -> shared.TrialSlopeRecipe:
    return shared.TrialSlopeRecipe(
        preset="regular",
        bids_root=Path("dummy"),
        subject=None,
        trial_slope_output_description="onset",
        hilbert_output_description="bga",
        hilbert_smoothing_window_ms_for_stats=250,
        hilbert_output_format=output_format,
        enable_hilbert_notch_filter=False,
        hilbert_notch_filter_freqs=[],
        use_hfo_spike_event_filter=False,
        hfo_spike_event_filter_rois=[],
        hfo_spike_event_filter_mode="channel",
        anchor_event_codes=["11", "12"],
        experiment_start_event_code="5",
        experiment_end_event_code=None,
        epoch_tmin_s=-1.0,
        epoch_tmax_s=6.0,
        enable_trial_stats_notch_filter=False,
        trial_stats_notch_filter_freqs=[],
        hilbert_file_filters={"suffix": "ieeg", "extension": ".mat"},
        hilbert_secondary_filters=[],
        hilbert_smoothing_windows_ms=[0, 250],
        trial_slope_ieeg_filters={"suffix": "ieeg", "extension": ".mat"},
        trial_slope_secondary_filters=[],
        roi_csv_files={},
        group_roi_combinations={},
        group_keep_combined_source_rois=True,
        group_param_kwargs={},
        use_matlab_zscores=False,
        matlab_zscores_path=Path("dummy.mat"),
        regression_output_format="matlab",
        regression_include_epochs=False,
        regression_group_output_format="matlab",
    )


def test_trial_slope_input_desc_uses_base_hilbert_desc_for_matlab() -> None:
    recipe = _make_recipe("matlab")

    assert recipe.hilbert_derivative_description == "bga"
    assert recipe.trial_slope_primary_filters()["desc"] == "bga"


def test_trial_slope_input_desc_uses_base_hilbert_desc_for_hdf5() -> None:
    recipe = _make_recipe("hdf5")

    assert recipe.hilbert_derivative_description == "bga"
    assert recipe.trial_slope_primary_filters()["desc"] == "bga"


def test_trial_slope_input_desc_includes_smoothing_for_brainvision() -> None:
    recipe = _make_recipe("brainvision")

    assert recipe.hilbert_derivative_description == "bgasm250"
    assert recipe.trial_slope_primary_filters()["desc"] == "bgasm250"

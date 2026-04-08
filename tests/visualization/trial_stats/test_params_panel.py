"""Smoke tests for ParamsPanel — set_params / get_params round-trip."""

from __future__ import annotations

import pytest

from gin_bids_py_analysis.processing.trial_slope_stats import TrialSlopeStatsParams
from gin_bids_py_analysis.processing.trial_stats import TrialStatsParams
from gin_bids_py_analysis.visualization.trial_stats.panels.params_panel import ParamsPanel


class TestParamsPanelRoundTrip:
    def test_default_params_round_trip(self, qtbot, default_params):
        panel = ParamsPanel(default_params)
        qtbot.addWidget(panel)

        recovered = panel.get_params()

        assert recovered.anchor_event_codes == default_params.anchor_event_codes
        assert recovered.tmin_s == pytest.approx(default_params.tmin_s)
        assert recovered.tmax_s == pytest.approx(default_params.tmax_s)
        assert recovered.condition_a == default_params.condition_a
        assert recovered.condition_b == default_params.condition_b
        assert recovered.p_value_correction_method == default_params.p_value_correction_method
        assert recovered.significance_alpha == pytest.approx(default_params.significance_alpha)
        assert recovered.min_trials_per_condition == default_params.min_trials_per_condition
        assert recovered.drop_partial_epochs == default_params.drop_partial_epochs
        assert recovered.equal_var == default_params.equal_var
        assert recovered.atlas_name is None
        assert recovered.atlas_regions == []
        assert recovered.window_ms == pytest.approx(0.0)
        assert recovered.n_bins == 0
        assert recovered.activity_scaling == "none"
        assert recovered.activity_baseline_tmin_s == pytest.approx(-0.2)
        assert recovered.activity_baseline_tmax_s == pytest.approx(0.0)

    def test_set_then_get(self, qtbot, default_params):
        panel = ParamsPanel(default_params)
        qtbot.addWidget(panel)

        new_params = TrialStatsParams(
            anchor_event_codes=["5", "15"],
            tmin_s=-0.5,
            tmax_s=3.0,
            condition_a="go",
            condition_b="nogo",
            min_trials_per_condition=5,
            drop_partial_epochs=False,
            equal_var=True,
            p_value_correction_method="bonferroni",
            significance_alpha=0.01,
            atlas_name=None,
            atlas_regions=[],
            window_ms=0.0,
            n_bins=0,
            activity_scaling="zscore_by_baseline",
            activity_baseline_tmin_s=-0.1,
            activity_baseline_tmax_s=0.0,
        )
        panel.set_params(new_params)
        recovered = panel.get_params()

        assert recovered.anchor_event_codes == ["5", "15"]
        assert recovered.tmin_s == pytest.approx(-0.5)
        assert recovered.tmax_s == pytest.approx(3.0)
        assert recovered.condition_a == "go"
        assert recovered.condition_b == "nogo"
        assert recovered.min_trials_per_condition == 5
        assert not recovered.drop_partial_epochs
        assert recovered.equal_var
        assert recovered.p_value_correction_method == "bonferroni"
        assert recovered.significance_alpha == pytest.approx(0.01)
        assert recovered.activity_scaling == "zscore_by_baseline"
        assert recovered.activity_baseline_tmin_s == pytest.approx(-0.1)
        assert recovered.activity_baseline_tmax_s == pytest.approx(0.0)

    def test_atlas_fields_round_trip(self, qtbot, default_params):
        panel = ParamsPanel(default_params)
        qtbot.addWidget(panel)

        params_with_atlas = TrialStatsParams(
            anchor_event_codes=["10"],
            tmin_s=-1.0,
            tmax_s=2.0,
            condition_a="accepted",
            condition_b="rejected",
            atlas_name="MarsAtlas",
            atlas_regions=["frontal", "parietal"],
        )
        panel.set_params(params_with_atlas)
        recovered = panel.get_params()

        assert recovered.atlas_name == "MarsAtlas"
        assert recovered.atlas_regions == ["frontal", "parietal"]

    def test_window_ms_n_bins_mutual_exclusion(self, qtbot, default_params):
        panel = ParamsPanel(default_params)
        qtbot.addWidget(panel)

        # Set window_ms first
        panel._window_ms.setValue(50.0)
        panel._n_bins.setValue(10)  # should clear window_ms
        assert panel._window_ms.value() == pytest.approx(0.0)

        # Set n_bins first
        panel._n_bins.setValue(20)
        panel._window_ms.setValue(100.0)  # should clear n_bins
        assert panel._n_bins.value() == 0

    def test_set_status(self, qtbot, default_params):
        panel = ParamsPanel(default_params)
        qtbot.addWidget(panel)
        panel.set_status("Test message")
        assert panel._status_label.text() == "Test message"

    def test_set_computing_disables_button(self, qtbot, default_params):
        panel = ParamsPanel(default_params)
        qtbot.addWidget(panel)

        panel.set_computing(True)
        assert not panel._compute_btn.isEnabled()
        assert "Computing" in panel._status_label.text()

        panel.set_computing(False)
        assert panel._compute_btn.isEnabled()

    def test_slope_mode_round_trip(self, qtbot, default_params, default_slope_params):
        panel = ParamsPanel(
            default_params,
            slope_params=default_slope_params,
            default_mode="slope",
        )
        qtbot.addWidget(panel)

        mode, params = panel.get_mode_and_params()
        assert mode == "slope"
        assert isinstance(params, TrialSlopeStatsParams)
        assert params.predictor == default_slope_params.predictor

    def test_slope_mode_round_trip_preserves_activity_scaling(self, qtbot, default_params, default_slope_params):
        slope_params = default_slope_params.model_copy(
            update={
                "activity_scaling": "zscore_by_baseline",
                "activity_baseline_tmin_s": -0.1,
                "activity_baseline_tmax_s": 0.0,
            }
        )
        panel = ParamsPanel(
            default_params,
            slope_params=slope_params,
            default_mode="slope",
        )
        qtbot.addWidget(panel)

        mode, params = panel.get_mode_and_params()
        assert mode == "slope"
        assert isinstance(params, TrialSlopeStatsParams)
        assert params.activity_scaling == "zscore_by_baseline"
        assert params.activity_baseline_tmin_s == pytest.approx(-0.1)
        assert params.activity_baseline_tmax_s == pytest.approx(0.0)

    def test_ttest_round_trip_preserves_script_only_fields(self, qtbot, default_params):
        params = default_params.model_copy(
            update={
                "n_permutations": 250,
                "permutation_seed": 123,
                "experiment_start_event_code": "EXP_START",
                "experiment_end_event_code": "EXP_END",
            }
        )
        panel = ParamsPanel(params)
        qtbot.addWidget(panel)

        recovered = panel.get_params()

        assert recovered.n_permutations == 250
        assert recovered.permutation_seed == 123
        assert recovered.experiment_start_event_code == "EXP_START"
        assert recovered.experiment_end_event_code == "EXP_END"

    def test_slope_round_trip_preserves_script_only_fields(self, qtbot, default_params, default_slope_params):
        slope_params = TrialSlopeStatsParams(
            **(
                default_slope_params.model_dump()
                | {
                    "predictor_transform_by_condition": {
                        "accepted": {"scale": 1.0, "offset": 0.0},
                        "rejected": {"scale": -1.0, "offset": 0.5},
                    },
                    "experiment_start_event_code": "EXP_START",
                    "experiment_end_event_code": "EXP_END",
                }
            )
        )
        panel = ParamsPanel(
            default_params,
            slope_params=slope_params,
            default_mode="slope",
        )
        qtbot.addWidget(panel)

        mode, params = panel.get_mode_and_params()

        assert mode == "slope"
        assert isinstance(params, TrialSlopeStatsParams)
        assert params.predictor_transform_by_condition["rejected"].scale == pytest.approx(-1.0)
        assert params.predictor_transform_by_condition["rejected"].offset == pytest.approx(0.5)
        assert params.experiment_start_event_code == "EXP_START"
        assert params.experiment_end_event_code == "EXP_END"



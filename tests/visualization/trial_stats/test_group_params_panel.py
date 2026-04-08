"""Smoke tests for GroupParamsPanel — set_params / get_params round-trip."""

from __future__ import annotations

import pytest

from gin_bids_py_analysis.processing.trial_stats_group.params import TrialStatsGroupParams
from gin_bids_py_analysis.visualization.trial_stats.panels.group_params_panel import (
    GroupParamsPanel,
)


@pytest.fixture()
def default_group_params() -> TrialStatsGroupParams:
    return TrialStatsGroupParams(
        source_metric="t_values",
        p_value_correction_method="fdr_bh",
        cluster_permutation_method="custom",
        significance_alpha=0.05,
        roi_mode="manual",
        manual_region_channels={
            "regionA": {"01": ["CH1", "CH2"], "02": ["CH3"]},
        },
        min_channels_per_roi=1,
        min_subjects_per_roi=1,
    )


class TestGroupParamsPanelRoundTrip:
    def test_default_params_round_trip(self, qtbot, default_group_params):
        panel = GroupParamsPanel(default_group_params)
        qtbot.addWidget(panel)

        recovered = panel.get_params()

        assert recovered.source_metric == default_group_params.source_metric
        assert recovered.p_value_correction_method == default_group_params.p_value_correction_method
        assert recovered.cluster_permutation_method == default_group_params.cluster_permutation_method
        assert recovered.significance_alpha == pytest.approx(default_group_params.significance_alpha)
        assert recovered.roi_mode == default_group_params.roi_mode
        assert recovered.manual_region_channels == default_group_params.manual_region_channels
        assert recovered.min_channels_per_roi == default_group_params.min_channels_per_roi
        assert recovered.min_subjects_per_roi == default_group_params.min_subjects_per_roi

    def test_set_then_get(self, qtbot, default_group_params):
        panel = GroupParamsPanel(default_group_params)
        qtbot.addWidget(panel)

        new_params = TrialStatsGroupParams(
            source_metric="mean_difference",
            p_value_correction_method="bonferroni",
            cluster_permutation_method="mne",
            significance_alpha=0.01,
            roi_mode="manual",
            manual_region_channels={"regionB": {"03": ["CH5"]}},
            min_channels_per_roi=2,
            min_subjects_per_roi=3,
        )
        panel.set_params(new_params)
        recovered = panel.get_params()

        assert recovered.source_metric == "mean_difference"
        assert recovered.p_value_correction_method == "bonferroni"
        assert recovered.cluster_permutation_method == "mne"
        assert recovered.significance_alpha == pytest.approx(0.01)
        assert recovered.manual_region_channels == new_params.manual_region_channels
        assert recovered.min_channels_per_roi == 2
        assert recovered.min_subjects_per_roi == 3

    def test_set_computing_disables_button(self, qtbot, default_group_params):
        panel = GroupParamsPanel(default_group_params)
        qtbot.addWidget(panel)

        panel.set_computing(True)
        assert not panel._compute_btn.isEnabled()
        assert panel._compute_btn.text() == "Computing…"

        panel.set_computing(False)
        assert panel._compute_btn.isEnabled()
        assert panel._compute_btn.text() == "Compute group stats"

    def test_set_status_updates_label(self, qtbot, default_group_params):
        panel = GroupParamsPanel(default_group_params)
        qtbot.addWidget(panel)

        panel.set_status("Test status message")
        assert panel._status_label.text() == "Test status message"

    def test_round_trip_preserves_script_only_ttest_fields(self, qtbot, default_group_params):
        params = default_group_params.model_copy(
            update={
                "n_group_permutations": 500,
                "cluster_threshold_alpha": 0.01,
                "permutation_seed": 321,
            }
        )
        panel = GroupParamsPanel(params)
        qtbot.addWidget(panel)

        recovered = panel.get_params()

        assert recovered.n_group_permutations == 500
        assert recovered.cluster_threshold_alpha == pytest.approx(0.01)
        assert recovered.permutation_seed == 321

    def test_compute_requested_signal_emitted(self, qtbot, default_group_params):
        panel = GroupParamsPanel(default_group_params)
        qtbot.addWidget(panel)

        with qtbot.waitSignal(panel.compute_requested, timeout=1000):
            panel._compute_btn.click()

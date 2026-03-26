"""Smoke tests for GroupPlotPanel — placeholder and update_plots."""

from __future__ import annotations

import numpy as np
import pytest

from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_stats_group.result import (
    TrialStatsGroupProcessingResult,
)
from gin_bids_py_analysis.visualization.trial_stats.panels.group_plot_panel import (
    GroupPlotPanel,
)


@pytest.fixture()
def synthetic_group_result(synthetic_result) -> TrialStatsGroupProcessingResult:
    """Minimal TrialStatsGroupProcessingResult with 3 ROIs and 60 time points."""
    rng = np.random.default_rng(seed=0)
    n_roi, n_t = 3, 60

    time_axis = np.linspace(-1.0, 2.0, n_t)
    t_vals = rng.standard_normal((n_roi, n_t))
    p_values = rng.uniform(0.001, 0.1, (n_roi, n_t))
    p_corr = np.clip(p_values, 0, 1)
    sig_mask = p_corr < 0.05
    metric_mean = rng.standard_normal((n_roi, n_t))
    metric_sem = np.abs(rng.standard_normal((n_roi, n_t))) * 0.1

    epoch_t = rng.standard_normal(n_roi)
    epoch_p = rng.uniform(0, 0.1, n_roi)
    epoch_mean = rng.standard_normal(n_roi)
    epoch_sem = np.abs(rng.standard_normal(n_roi)) * 0.1
    epoch_df = np.full(n_roi, 10.0)

    return TrialStatsGroupProcessingResult(
        source_group=BIDSFileGroup(primary=synthetic_result.source_group.primary),
        metadata={},
        output_entities=None,
        t_values=t_vals,
        p_values=p_corr,
        p_values_uncorrected=p_values,
        significant_mask=sig_mask,
        metric_mean=metric_mean,
        metric_sem=metric_sem,
        time_axis_s=time_axis,
        region_names=["regionA", "regionB", "regionC"],
        source_metric="t_values",
        condition_labels=("accepted", "rejected"),
        p_value_correction_method="fdr_bh",
        significance_alpha=0.05,
        roi_mode="manual",
        epoch_mean_t_values=epoch_t,
        epoch_mean_p_values=epoch_p,
        epoch_mean_df=epoch_df,
        epoch_mean_metric_mean=epoch_mean,
        epoch_mean_metric_sem=epoch_sem,
        roi_channel_counts=np.array([2, 3, 1]),
        roi_subject_counts=np.array([2, 2, 1]),
        contributions=[],
        source_trial_stats_files=[],
        source_electrodes_files=[],
        excluded_rois={},
    )


class TestGroupPlotPanelPlaceholder:
    def test_placeholder_renders_without_error(self, qtbot):
        panel = GroupPlotPanel()
        qtbot.addWidget(panel)
        panel.show_placeholder()
        # No exception == pass

    def test_initial_state_has_roi_list(self, qtbot):
        panel = GroupPlotPanel()
        qtbot.addWidget(panel)
        # ROI list widget should exist and be empty initially
        assert panel._roi_list.count() == 0


class TestGroupPlotPanelUpdatePlots:
    def test_update_plots_populates_roi_list(self, qtbot, synthetic_group_result):
        panel = GroupPlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_group_result, 0)

        assert panel._roi_list.count() == 3
        assert panel._roi_list.item(0).text() == "regionA"
        assert panel._roi_list.item(1).text() == "regionB"
        assert panel._roi_list.item(2).text() == "regionC"

    def test_current_roi_index_after_update(self, qtbot, synthetic_group_result):
        panel = GroupPlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_group_result, 0)

        assert panel.current_roi_index == 0

    def test_current_roi_name(self, qtbot, synthetic_group_result):
        panel = GroupPlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_group_result, 0)

        assert panel.current_roi_name == "regionA"

    def test_roi_changed_signal_on_selection_change(self, qtbot, synthetic_group_result):
        panel = GroupPlotPanel()
        qtbot.addWidget(panel)
        panel.update_plots(synthetic_group_result, 0)

        with qtbot.waitSignal(panel.roi_changed, timeout=1000) as blocker:
            panel._roi_list.setCurrentRow(2)

        assert blocker.args[0] == 2

    def test_update_to_second_roi(self, qtbot, synthetic_group_result):
        panel = GroupPlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_group_result, 1)

        assert panel.current_roi_index == 1
        assert panel.current_roi_name == "regionB"

"""Smoke tests for GroupPlotPanel — placeholder and update_plots."""

from __future__ import annotations

import numpy as np
import pytest

from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_slope_stats_group.result import (
    TrialSlopeStatsGroupProcessingResult,
)
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


@pytest.fixture()
def synthetic_slope_group_result(synthetic_result) -> TrialSlopeStatsGroupProcessingResult:
    n_roi, n_t = 2, 60
    time_axis = np.linspace(-1.0, 2.0, n_t)
    shape = (n_roi, n_t)
    cond_labels = ("pleasant", "unpleasant")

    return TrialSlopeStatsGroupProcessingResult(
        source_group=BIDSFileGroup(primary=synthetic_result.source_group.primary),
        metadata={},
        output_entities=None,
        condition_a_slope_t_values=np.full(shape, 2.0, dtype=np.float64),
        condition_a_slope_p_values=np.full(shape, 0.02, dtype=np.float64),
        condition_a_slope_p_values_uncorrected=np.full(shape, 0.03, dtype=np.float64),
        condition_a_slope_significant_mask=np.ones(shape, dtype=bool),
        condition_b_slope_t_values=np.full(shape, -1.5, dtype=np.float64),
        condition_b_slope_p_values=np.full(shape, 0.04, dtype=np.float64),
        condition_b_slope_p_values_uncorrected=np.full(shape, 0.05, dtype=np.float64),
        condition_b_slope_significant_mask=np.zeros(shape, dtype=bool),
        condition_a_slope_mean=np.full(shape, 0.8, dtype=np.float64),
        condition_a_slope_sem=np.full(shape, 0.1, dtype=np.float64),
        condition_b_slope_mean=np.full(shape, -0.6, dtype=np.float64),
        condition_b_slope_sem=np.full(shape, 0.1, dtype=np.float64),
        condition_a_activity_mean=np.full(shape, 1.0, dtype=np.float64),
        condition_a_activity_sem=np.full(shape, 0.2, dtype=np.float64),
        condition_b_activity_mean=np.full(shape, 0.8, dtype=np.float64),
        condition_b_activity_sem=np.full(shape, 0.2, dtype=np.float64),
        condition_a_r_value_mean=np.full(shape, 0.2, dtype=np.float64),
        condition_a_r_value_sem=np.full(shape, 0.05, dtype=np.float64),
        condition_b_r_value_mean=np.full(shape, -0.1, dtype=np.float64),
        condition_b_r_value_sem=np.full(shape, 0.05, dtype=np.float64),
        condition_a_epoch_slope_t=np.full(n_roi, 2.0, dtype=np.float64),
        condition_a_epoch_slope_p=np.full(n_roi, 0.02, dtype=np.float64),
        condition_a_epoch_slope_df=np.full(n_roi, 10.0, dtype=np.float64),
        condition_a_epoch_slope_mean=np.full(n_roi, 0.8, dtype=np.float64),
        condition_a_epoch_slope_sem=np.full(n_roi, 0.1, dtype=np.float64),
        condition_b_epoch_slope_t=np.full(n_roi, -1.5, dtype=np.float64),
        condition_b_epoch_slope_p=np.full(n_roi, 0.04, dtype=np.float64),
        condition_b_epoch_slope_df=np.full(n_roi, 10.0, dtype=np.float64),
        condition_b_epoch_slope_mean=np.full(n_roi, -0.6, dtype=np.float64),
        condition_b_epoch_slope_sem=np.full(n_roi, 0.1, dtype=np.float64),
        time_axis_s=time_axis,
        region_names=["slope_roi_a", "slope_roi_b"],
        condition_labels=cond_labels,
        roi_channel_counts=np.array([4, 3], dtype=np.int64),
        roi_subject_counts=np.array([2, 2], dtype=np.int64),
        contributions=[],
        condition_a_slope_contributions=[
            np.ones((3, n_t), dtype=np.float64),
            np.ones((2, n_t), dtype=np.float64),
        ],
        condition_b_slope_contributions=[
            -np.ones((3, n_t), dtype=np.float64),
            -np.ones((2, n_t), dtype=np.float64),
        ],
        condition_a_activity_contributions=[
            np.full((3, n_t), 1.0, dtype=np.float64),
            np.full((2, n_t), 0.9, dtype=np.float64),
        ],
        condition_b_activity_contributions=[
            np.full((3, n_t), 0.8, dtype=np.float64),
            np.full((2, n_t), 0.7, dtype=np.float64),
        ],
        contribution_labels=[
            ["01/A1", "01/A2", "02/A1"],
            ["01/B1", "02/B1"],
        ],
        p_value_correction_method="none",
        significance_alpha=0.05,
        roi_mode="manual",
        atlas_name=None,
        source_trial_slope_stats_files=[],
        source_electrodes_files=[],
        excluded_rois={},
    )


class TestGroupPlotPanelSlopeUpdate:
    def test_update_plots_with_slope_group_result(self, qtbot, synthetic_slope_group_result):
        panel = GroupPlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_slope_group_result, 0)

        assert panel._roi_list.count() == 2
        assert panel.current_roi_name == "slope_roi_a"
        assert panel._plot_tabs.tabText(0) == "Activity"
        assert panel._plot_tabs.tabText(1) == "Slope mean"

    def test_activity_means_are_plotted_in_activity_axis(self, qtbot, synthetic_slope_group_result):
        panel = GroupPlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_slope_group_result, 0)

        labeled_lines = {
            line.get_label(): line
            for line in panel._ax_means.lines
            if line.get_label() in synthetic_slope_group_result.condition_labels
        }
        assert set(labeled_lines) == set(synthetic_slope_group_result.condition_labels)
        assert np.allclose(labeled_lines["pleasant"].get_ydata(), 1.0)
        assert np.allclose(labeled_lines["unpleasant"].get_ydata(), 0.8)

    def test_slope_means_are_plotted_in_slope_axis(self, qtbot, synthetic_slope_group_result):
        panel = GroupPlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_slope_group_result, 0)

        labeled_lines = {
            line.get_label(): line
            for line in panel._ax_slope.lines
            if line.get_label() in synthetic_slope_group_result.condition_labels
        }
        assert set(labeled_lines) == set(synthetic_slope_group_result.condition_labels)
        assert np.allclose(labeled_lines["pleasant"].get_ydata(), 0.8)
        assert np.allclose(labeled_lines["unpleasant"].get_ydata(), -0.6)

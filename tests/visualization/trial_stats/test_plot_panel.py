"""Smoke tests for PlotPanel — update_plots and placeholder rendering."""

from __future__ import annotations

import numpy as np
import pytest

from gin_bids_py_analysis.visualization.trial_stats.panels.plot_panel import (
    PlotPanel,
    _compute_scatter_summary_points,
    _predictor_axis_label,
)


class TestPlotPanel:
    def test_creates_without_error(self, qtbot):
        panel = PlotPanel()
        qtbot.addWidget(panel)

    def test_update_plots_valid_result(self, qtbot, synthetic_result):
        panel = PlotPanel()
        qtbot.addWidget(panel)
        # Should not raise for any valid channel index
        panel.update_plots(synthetic_result, channel_idx=0)
        panel.update_plots(synthetic_result, channel_idx=3)

    def test_update_plots_all_channels(self, qtbot, synthetic_result):
        panel = PlotPanel()
        qtbot.addWidget(panel)
        for ch in range(len(synthetic_result.channel_names)):
            panel.update_plots(synthetic_result, channel_idx=ch)

    def test_placeholder_on_invalid_stats(self, qtbot, synthetic_result):
        """When stats_valid=False the plot panel shows the placeholder instead of crashing."""
        synthetic_result.stats_valid = False
        panel = PlotPanel()
        qtbot.addWidget(panel)
        # Must not raise
        panel.update_plots(synthetic_result, channel_idx=0)

    def test_show_placeholder(self, qtbot):
        panel = PlotPanel()
        qtbot.addWidget(panel)
        panel.show_placeholder()  # Must not raise

    def test_means_plot_has_two_lines(self, qtbot, synthetic_result):
        panel = PlotPanel()
        qtbot.addWidget(panel)
        panel.update_plots(synthetic_result, channel_idx=0)
        # Two condition mean lines + one axvline(0)
        lines = panel._ax_means.get_lines()
        assert len(lines) == 3

    def test_t_plot_has_one_line(self, qtbot, synthetic_result):
        panel = PlotPanel()
        qtbot.addWidget(panel)
        panel.update_plots(synthetic_result, channel_idx=0)
        # t-value line + axhline(0) + axvline(0)
        lines = panel._ax_t.get_lines()
        assert len(lines) == 3  # t-value + axhline + axvline

    def test_p_plot_y_limits_non_negative(self, qtbot, synthetic_result):
        panel = PlotPanel()
        qtbot.addWidget(panel)
        panel.update_plots(synthetic_result, channel_idx=0)
        ymin, _ = panel._ax_p.get_ylim()
        assert ymin == pytest.approx(0.0)

    def test_update_plots_slope_mode(self, qtbot, synthetic_slope_result):
        panel = PlotPanel()
        qtbot.addWidget(panel)
        panel.update_plots(synthetic_slope_result, channel_idx=0)
        assert panel._tabs.tabText(1) == "Slopes"
        assert panel._tabs.isTabEnabled(panel._scatter_tab_index)

    def test_ttest_axes_have_titles(self, qtbot, synthetic_result):
        panel = PlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_result, channel_idx=0)

        assert "t-values" in panel._ax_t.get_title()
        assert "p-values" in panel._ax_p.get_title()
        assert not panel._tabs.isTabEnabled(panel._scatter_tab_index)

    def test_slope_axes_have_titles(self, qtbot, synthetic_slope_result):
        panel = PlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_slope_result, channel_idx=0)

        assert "slopes" in panel._ax_t.get_title()
        assert "slope p-values" in panel._ax_p.get_title()

    def test_scatter_tab_is_present(self, qtbot):
        panel = PlotPanel()
        qtbot.addWidget(panel)

        assert panel._tabs.tabText(panel._scatter_tab_index) == "Scatter"

    def test_slope_scatter_draws_points_and_regressions(self, qtbot, synthetic_slope_result):
        panel = PlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_slope_result, channel_idx=0)

        assert len(panel._ax_scatter.collections) >= 2
        assert len(panel._ax_scatter.lines) >= 2
        assert len(panel._ax_scatter.containers) == 2
        assert "predictor vs activity" in panel._ax_scatter.get_title()
        assert panel._ax_scatter.get_xlabel() == "predictor_value"
        assert panel._ax_scatter.get_ylabel() == "Epoch mean activity"

    def test_compute_scatter_summary_points_builds_binned_means_and_sem(self):
        predictor = np.linspace(-50.0, 50.0, 10)
        activity = np.linspace(0.0, 1.0, 10)

        mean_x, mean_y, sem_x, sem_y = _compute_scatter_summary_points(
            predictor,
            activity,
            target_bins=5,
        )

        assert mean_x.shape == (5,)
        assert mean_y.shape == (5,)
        assert sem_x.shape == (5,)
        assert sem_y.shape == (5,)
        assert np.all(np.diff(mean_x) > 0.0)
        assert np.all(np.diff(mean_y) > 0.0)
        assert np.all(sem_x >= 0.0)
        assert np.all(sem_y >= 0.0)

    def test_slope_scatter_uses_zscore_ylabel_when_activity_is_scaled(self, qtbot, synthetic_slope_result):
        synthetic_slope_result.activity_zscore = "baseline"
        panel = PlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_slope_result, channel_idx=0)

        assert panel._ax_scatter.get_ylabel() == "Epoch mean activity (z)"

    def test_trial_matrix_hides_rows_that_are_all_nan_for_selected_feature(
        self,
        qtbot,
        synthetic_slope_result,
    ):
        n_t = synthetic_slope_result.time_axis_s.size
        epochs_a = np.ones((3, 4, n_t), dtype=np.float64)
        epochs_b = np.ones((2, 4, n_t), dtype=np.float64)
        epochs_a[1, 0, :] = np.nan
        epochs_b[0, 0, :] = np.nan
        synthetic_slope_result.condition_a_epochs = epochs_a
        synthetic_slope_result.condition_b_epochs = epochs_b

        panel = PlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_slope_result, channel_idx=0)

        assert panel._ax_matrix.images[0].get_array().shape == (3, n_t)
        assert panel._ax_matrix.get_title() == "A1 - trials shown (2 / 1), masked (1 / 1)"

    def test_trial_matrix_shows_placeholder_when_selected_feature_has_no_finite_rows(
        self,
        qtbot,
        synthetic_slope_result,
    ):
        n_t = synthetic_slope_result.time_axis_s.size
        epochs_a = np.ones((2, 4, n_t), dtype=np.float64)
        epochs_b = np.ones((1, 4, n_t), dtype=np.float64)
        epochs_a[:, 0, :] = np.nan
        epochs_b[:, 0, :] = np.nan
        synthetic_slope_result.condition_a_epochs = epochs_a
        synthetic_slope_result.condition_b_epochs = epochs_b

        panel = PlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_slope_result, channel_idx=0)

        assert len(panel._ax_matrix.images) == 0
        assert panel._ax_matrix.texts[0].get_text() == "No finite trial data for selected feature"

    def test_predictor_axis_label_marks_condition_and_global_zscore(self):
        assert _predictor_axis_label("rating", "condition") == "rating (z)"
        assert _predictor_axis_label("rating", "global") == "rating (z)"
        assert _predictor_axis_label("rating", "none") == "rating"

    def test_slope_scatter_uses_dashed_regression_when_not_significant(
        self,
        qtbot,
        synthetic_slope_result,
    ):
        synthetic_slope_result.condition_a_predictor_values = np.arange(12, dtype=np.float64)
        synthetic_slope_result.condition_b_predictor_values = np.arange(11, dtype=np.float64)
        synthetic_slope_result.condition_a_trial_activity_summary_values[0] = np.array(
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            dtype=np.float64,
        )
        synthetic_slope_result.condition_b_trial_activity_summary_values[0] = np.array(
            [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            dtype=np.float64,
        )
        panel = PlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_slope_result, channel_idx=0)

        assert any(
            line.get_linestyle() == "--" for line in panel._ax_scatter.lines
        )

    def test_slope_scatter_shows_placeholder_when_epoch_means_missing(
        self,
        qtbot,
        synthetic_slope_result,
    ):
        synthetic_slope_result.condition_a_trial_activity_summary_values = np.array([])
        synthetic_slope_result.condition_b_trial_activity_summary_values = np.array([])
        panel = PlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_slope_result, channel_idx=0)

        assert panel._ax_scatter.texts[0].get_text() == "No scatter data available"

    def test_slope_scatter_shows_placeholder_when_epoch_means_are_incoherent(
        self,
        qtbot,
        synthetic_slope_result,
    ):
        synthetic_slope_result.condition_b_trial_activity_summary_values = np.ones((4, 10), dtype=np.float64)
        panel = PlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_slope_result, channel_idx=0)

        assert panel._ax_scatter.texts[0].get_text() == "No scatter data available"

    def test_slope_scatter_uses_trial_activity_summary_label(
        self,
        qtbot,
        synthetic_slope_result,
    ):
        synthetic_slope_result.trial_activity_summary_kind = "anchor_to_response_mean"
        synthetic_slope_result.trial_activity_summary_label = "Mean activity (trigger to response)"
        panel = PlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_slope_result, channel_idx=0)

        assert panel._ax_scatter.get_ylabel() == "Mean activity (trigger to response)"

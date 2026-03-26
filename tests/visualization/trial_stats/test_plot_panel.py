"""Smoke tests for PlotPanel — update_plots and placeholder rendering."""

from __future__ import annotations

import numpy as np
import pytest

from gin_bids_py_analysis.visualization.trial_stats.panels.plot_panel import PlotPanel


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

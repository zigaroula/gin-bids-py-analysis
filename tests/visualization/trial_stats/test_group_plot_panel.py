"""Smoke tests for GroupPlotPanel — placeholder and update_plots."""

from __future__ import annotations

import numpy as np
import pytest

from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_stats_group import (
    RegressionGroupProcessingResult,
)
from gin_bids_py_analysis.processing.trial_stats_group import (
    ConditionTestGroupProcessingResult,
    ConditionTestEpochSummary,
    GroupEpochStats,
    GroupEstimate,
    GroupEstimatePair,
    GroupTimecourseStats,
    IndexedConditionContributions,
    RegressionMetricStats,
    VsZeroStatsPair,
)
from gin_bids_py_analysis.visualization.trial_stats.panels.group_plot_panel import (
    GroupPlotPanel,
)


@pytest.fixture()
def synthetic_group_result(synthetic_result) -> ConditionTestGroupProcessingResult:
    """Minimal ConditionTestGroupProcessingResult with 3 ROIs and 60 time points."""
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

    return ConditionTestGroupProcessingResult(
        source_group=BIDSFileGroup(primary=synthetic_result.source_group.primary),
        metadata={},
        output_entities=None,
        signal_activity_stats=GroupTimecourseStats(
            t_values=t_vals,
            p_values=p_corr,
            p_values_uncorrected=p_values,
            significant_mask=sig_mask,
        ),
        condition_difference=GroupEstimate(mean=metric_mean, sem=metric_sem),
        time_axis_s=time_axis,
        region_names=["regionA", "regionB", "regionC"],
        primary_condition_metric="t_values",
        condition_labels=("accepted", "rejected"),
        p_value_correction_method="fdr_bh",
        significance_alpha=0.05,
        roi_mode="manual",
        signal_activity_epoch=GroupEpochStats(t=epoch_t, p=epoch_p, df=epoch_df),
        summary_epoch=ConditionTestEpochSummary(
            t_values=epoch_t,
            p_values=epoch_p,
            df=epoch_df,
            condition_difference=GroupEstimate(mean=epoch_mean, sem=epoch_sem),
        ),
        roi_channel_counts=np.array([2, 3, 1]),
        roi_subject_counts=np.array([2, 2, 1]),
        contributions=[],
        signal_activity=GroupEstimatePair(
            condition_a=GroupEstimate(
                mean=np.full((n_roi, n_t), 1.0, dtype=np.float64),
                sem=np.full((n_roi, n_t), 0.2, dtype=np.float64),
            ),
            condition_b=GroupEstimate(
                mean=np.full((n_roi, n_t), 0.7, dtype=np.float64),
                sem=np.full((n_roi, n_t), 0.15, dtype=np.float64),
            ),
        ),
        source_subject_stats_files=[],
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

    def test_matrix_row_label_toggles_are_off_by_default(self, qtbot):
        panel = GroupPlotPanel()
        qtbot.addWidget(panel)

        assert panel._activity_matrix_row_labels_checkbox.isChecked() is False
        assert panel._matrix_row_labels_checkbox.isChecked() is False


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

    def test_ttest_group_stats_axes_have_titles(self, qtbot, synthetic_group_result):
        panel = GroupPlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_group_result, 0)

        assert "t-values" in panel._ax_t.get_title()
        assert "p-values" in panel._ax_p.get_title()


@pytest.fixture()
def synthetic_slope_group_result(synthetic_result) -> RegressionGroupProcessingResult:
    n_roi, n_t = 2, 60
    time_axis = np.linspace(-1.0, 2.0, n_t)
    shape = (n_roi, n_t)
    cond_labels = ("pleasant", "unpleasant")

    return RegressionGroupProcessingResult(
        source_group=BIDSFileGroup(primary=synthetic_result.source_group.primary),
        metadata={},
        output_entities=None,
        regression_stats=RegressionMetricStats(
            contrast=GroupTimecourseStats(
                t_values=np.full(shape, 2.0, dtype=np.float64),
                p_values=np.full(shape, 0.02, dtype=np.float64),
                p_values_uncorrected=np.full(shape, 0.03, dtype=np.float64),
                significant_mask=np.ones(shape, dtype=bool),
            ),
            epoch_summary=GroupEpochStats(
                t=np.full(n_roi, 2.0, dtype=np.float64),
                p=np.full(n_roi, 0.02, dtype=np.float64),
                df=np.full(n_roi, 10.0, dtype=np.float64),
            ),
            vs_zero=VsZeroStatsPair(
                condition_a=GroupTimecourseStats(
                    significant_mask=np.ones(shape, dtype=bool),
                ),
                condition_b=GroupTimecourseStats(
                    significant_mask=np.ones(shape, dtype=bool),
                ),
            ),
        ),
        signal_activity_stats=GroupTimecourseStats(
            t_values=np.full(shape, -1.5, dtype=np.float64),
            p_values=np.full(shape, 0.04, dtype=np.float64),
            p_values_uncorrected=np.full(shape, 0.05, dtype=np.float64),
            significant_mask=np.zeros(shape, dtype=bool),
        ),
        signal_activity_epoch=GroupEpochStats(
            t=np.full(n_roi, -1.5, dtype=np.float64),
            p=np.full(n_roi, 0.04, dtype=np.float64),
            df=np.full(n_roi, 10.0, dtype=np.float64),
        ),
        slope=GroupEstimatePair(
            condition_a=GroupEstimate(
                mean=np.full(shape, 0.8, dtype=np.float64),
                sem=np.full(shape, 0.1, dtype=np.float64),
            ),
            condition_b=GroupEstimate(
                mean=np.full(shape, -0.6, dtype=np.float64),
                sem=np.full(shape, 0.1, dtype=np.float64),
            ),
        ),
        signal_activity=GroupEstimatePair(
            condition_a=GroupEstimate(
                mean=np.full(shape, 1.0, dtype=np.float64),
                sem=np.full(shape, 0.2, dtype=np.float64),
            ),
            condition_b=GroupEstimate(
                mean=np.full(shape, 0.8, dtype=np.float64),
                sem=np.full(shape, 0.2, dtype=np.float64),
            ),
        ),
        r_value=GroupEstimatePair(
            condition_a=GroupEstimate(
                mean=np.full(shape, 0.2, dtype=np.float64),
                sem=np.full(shape, 0.05, dtype=np.float64),
            ),
            condition_b=GroupEstimate(
                mean=np.full(shape, -0.1, dtype=np.float64),
                sem=np.full(shape, 0.05, dtype=np.float64),
            ),
        ),
        time_axis_s=time_axis,
        region_names=["slope_roi_a", "slope_roi_b"],
        condition_labels=cond_labels,
        primary_regression_metric="slope",
        contrast_mode="paired",
        roi_channel_counts=np.array([4, 3], dtype=np.int64),
        roi_subject_counts=np.array([2, 2], dtype=np.int64),
        contributions=[],
        slope_contributions=IndexedConditionContributions(
            condition_a=[
                np.ones((3, n_t), dtype=np.float64),
                np.ones((2, n_t), dtype=np.float64),
            ],
            condition_b=[
                -np.ones((3, n_t), dtype=np.float64),
                -np.ones((2, n_t), dtype=np.float64),
            ],
            labels=[
                ["01/A1", "01/A2", "02/A1"],
                ["01/B1", "02/B1"],
            ],
        ),
        signal_activity_contributions=IndexedConditionContributions(
            condition_a=[
                np.full((3, n_t), 1.0, dtype=np.float64),
                np.full((2, n_t), 0.9, dtype=np.float64),
            ],
            condition_b=[
                np.full((3, n_t), 0.8, dtype=np.float64),
                np.full((2, n_t), 0.7, dtype=np.float64),
            ],
            labels=[
                ["01/A1", "01/A2", "02/A1"],
                ["01/B1", "02/B1"],
            ],
        ),
        p_value_correction_method="none",
        significance_alpha=0.05,
        roi_mode="manual",
        atlas_name=None,
        source_subject_stats_files=[],
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
        assert panel._plot_tabs.tabText(0) == "Activity mean"
        assert panel._plot_tabs.tabText(4) == "Slope mean"

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

    def test_slope_group_axes_have_titles_with_roi_context(self, qtbot, synthetic_slope_group_result):
        panel = GroupPlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_slope_group_result, 0)

        assert "slope mean" in panel._ax_slope.get_title()
        assert "channel(s)" in panel._ax_slope.get_title()
        assert "activity p-values" in panel._ax_activity_p.get_title()
        assert "slope p-values" in panel._ax_p.get_title()

    def test_slope_group_uses_zscore_activity_labels_when_metadata_requests_it(self, qtbot, synthetic_slope_group_result):
        synthetic_slope_group_result.metadata = {"activity_zscore": "baseline"}
        panel = GroupPlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_slope_group_result, 0)

        assert panel._ax_means.get_ylabel() == "mean region activity (z)"
        assert panel._ax_scatter.get_ylabel() == "Epoch mean activity (z)"

    def test_slope_matrix_row_labels_can_be_toggled_on_for_subject_channel_mapping(
        self,
        qtbot,
        synthetic_slope_group_result,
    ):
        panel = GroupPlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_slope_group_result, 0)

        assert [tick.get_text() for tick in panel._ax_matrix.get_yticklabels()] == [
            "pleasant",
            "unpleasant",
        ]

        panel._plot_tabs.setCurrentIndex(7)

        panel._matrix_row_labels_checkbox.setChecked(True)

        assert [tick.get_text() for tick in panel._ax_matrix.get_yticklabels()] == [
            "pleasant: sub-01 / A1",
            "pleasant: sub-01 / A2",
            "pleasant: sub-02 / A1",
            "unpleasant: sub-01 / A1",
            "unpleasant: sub-01 / A2",
            "unpleasant: sub-02 / A1",
        ]

    def test_activity_and_slope_matrix_toggles_have_independent_state(
        self,
        qtbot,
        synthetic_slope_group_result,
    ):
        panel = GroupPlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_slope_group_result, 0)

        panel._activity_matrix_row_labels_checkbox.setChecked(True)

        assert [tick.get_text() for tick in panel._ax_activity_matrix.get_yticklabels()] == [
            "pleasant: sub-01 / A1",
            "pleasant: sub-01 / A2",
            "pleasant: sub-02 / A1",
            "unpleasant: sub-01 / A1",
            "unpleasant: sub-01 / A2",
            "unpleasant: sub-02 / A1",
        ]
        assert [tick.get_text() for tick in panel._ax_matrix.get_yticklabels()] == [
            "pleasant",
            "unpleasant",
        ]

        panel._matrix_row_labels_checkbox.setChecked(True)

        assert [tick.get_text() for tick in panel._ax_matrix.get_yticklabels()] == [
            "pleasant: sub-01 / A1",
            "pleasant: sub-01 / A2",
            "pleasant: sub-02 / A1",
            "unpleasant: sub-01 / A1",
            "unpleasant: sub-01 / A2",
            "unpleasant: sub-02 / A1",
        ]

    def test_slope_matrix_hides_rows_that_are_all_nan(
        self,
        qtbot,
        synthetic_slope_group_result,
    ):
        synthetic_slope_group_result.slope_contributions.condition_a[0][1, :] = np.nan
        synthetic_slope_group_result.slope_contributions.condition_b[0][0, :] = np.nan
        panel = GroupPlotPanel()
        qtbot.addWidget(panel)

        panel.update_plots(synthetic_slope_group_result, 0)
        panel._matrix_row_labels_checkbox.setChecked(True)

        assert panel._ax_matrix.images[0].get_array().shape[0] == 4
        assert [tick.get_text() for tick in panel._ax_matrix.get_yticklabels()] == [
            "pleasant: sub-01 / A1",
            "pleasant: sub-02 / A1",
            "unpleasant: sub-01 / A2",
            "unpleasant: sub-02 / A1",
        ]




"""Group-level plot panel: ROI selector + tabbed matplotlib plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .plot_panel import (
    _SCATTER_SUMMARY_TARGET_BINS,
    _compute_scatter_summary_points,
    _fit_scatter_regression,
    _scatter_activity_axis_label,
)

if TYPE_CHECKING:
    from gin_bids_py_analysis.processing.trial_stats_group import (
        RegressionGroupProcessingResult,
    )
    from gin_bids_py_analysis.processing.trial_stats_group import (
        ConditionTestGroupProcessingResult,
    )


class GroupPlotPanel(QWidget):
    """Panel combining a ROI selector list and tabbed group plots.

    The panel supports two result schemas:
    - ``ConditionTestGroupProcessingResult`` (classic t-test group stats)
    - ``RegressionGroupProcessingResult`` (slope-regression group stats)
    """

    roi_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.addWidget(QLabel("ROIs"))
        self._roi_list = QListWidget()
        left_layout.addWidget(self._roi_list, stretch=1)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(0)
        right_layout.setContentsMargins(4, 4, 4, 4)
        self._plot_tabs = QTabWidget()

        activity_w = QWidget()
        activity_layout = QVBoxLayout(activity_w)
        activity_layout.setContentsMargins(0, 0, 0, 0)
        self._fig_means = Figure(tight_layout=True)
        self._ax_means = self._fig_means.add_subplot(111)
        self._canvas_means = FigureCanvasQTAgg(self._fig_means)
        activity_layout.addWidget(self._canvas_means)
        self._plot_tabs.addTab(activity_w, "Activity mean")

        activity_t_w = QWidget()
        activity_t_layout = QVBoxLayout(activity_t_w)
        activity_t_layout.setContentsMargins(0, 0, 0, 0)
        self._fig_activity_t = Figure(tight_layout=True)
        self._ax_activity_t = self._fig_activity_t.add_subplot(111)
        self._canvas_activity_t = FigureCanvasQTAgg(self._fig_activity_t)
        activity_t_layout.addWidget(self._canvas_activity_t)
        self._plot_tabs.addTab(activity_t_w, "Activity t-values")

        activity_p_w = QWidget()
        activity_p_layout = QVBoxLayout(activity_p_w)
        activity_p_layout.setContentsMargins(0, 0, 0, 0)
        self._fig_activity_p = Figure(tight_layout=True)
        self._ax_activity_p = self._fig_activity_p.add_subplot(111)
        self._canvas_activity_p = FigureCanvasQTAgg(self._fig_activity_p)
        activity_p_layout.addWidget(self._canvas_activity_p)
        self._plot_tabs.addTab(activity_p_w, "Activity p-values")

        activity_matrix_w = QWidget()
        activity_matrix_layout = QVBoxLayout(activity_matrix_w)
        activity_matrix_layout.setContentsMargins(0, 0, 0, 0)
        self._fig_activity_matrix = Figure(tight_layout=True)
        self._ax_activity_matrix = self._fig_activity_matrix.add_subplot(111)
        self._canvas_activity_matrix = FigureCanvasQTAgg(self._fig_activity_matrix)
        activity_matrix_layout.addWidget(self._canvas_activity_matrix)
        activity_matrix_controls_layout = QHBoxLayout()
        activity_matrix_controls_layout.setContentsMargins(0, 6, 0, 0)
        activity_matrix_controls_layout.addStretch(1)
        self._activity_matrix_row_labels_checkbox = QCheckBox("Show subject/channel row labels")
        self._activity_matrix_row_labels_checkbox.setToolTip(
            "Display one label per matrix row to identify the contributing subject and channel."
        )
        activity_matrix_controls_layout.addWidget(self._activity_matrix_row_labels_checkbox)
        activity_matrix_controls_layout.addStretch(1)
        activity_matrix_layout.addLayout(activity_matrix_controls_layout)
        self._plot_tabs.addTab(activity_matrix_w, "Activity matrix")

        slope_w = QWidget()
        slope_layout = QVBoxLayout(slope_w)
        slope_layout.setContentsMargins(0, 0, 0, 0)
        self._fig_slope = Figure(tight_layout=True)
        self._ax_slope = self._fig_slope.add_subplot(111)
        self._canvas_slope = FigureCanvasQTAgg(self._fig_slope)
        slope_layout.addWidget(self._canvas_slope)
        self._plot_tabs.addTab(slope_w, "Slope mean")

        t_w = QWidget()
        t_layout = QVBoxLayout(t_w)
        t_layout.setContentsMargins(0, 0, 0, 0)
        self._fig_t = Figure(tight_layout=True)
        self._ax_t = self._fig_t.add_subplot(111)
        self._canvas_t = FigureCanvasQTAgg(self._fig_t)
        t_layout.addWidget(self._canvas_t)
        self._plot_tabs.addTab(t_w, "Slope t-values")

        p_w = QWidget()
        p_layout = QVBoxLayout(p_w)
        p_layout.setContentsMargins(0, 0, 0, 0)
        self._fig_p = Figure(tight_layout=True)
        self._ax_p = self._fig_p.add_subplot(111)
        self._canvas_p = FigureCanvasQTAgg(self._fig_p)
        p_layout.addWidget(self._canvas_p)
        self._plot_tabs.addTab(p_w, "Slope p-values")

        matrix_w = QWidget()
        matrix_layout = QVBoxLayout(matrix_w)
        matrix_layout.setContentsMargins(0, 0, 0, 0)
        self._fig_matrix = Figure(tight_layout=True)
        self._ax_matrix = self._fig_matrix.add_subplot(111)
        self._canvas_matrix = FigureCanvasQTAgg(self._fig_matrix)
        matrix_layout.addWidget(self._canvas_matrix)
        matrix_controls_layout = QHBoxLayout()
        matrix_controls_layout.setContentsMargins(0, 6, 0, 0)
        matrix_controls_layout.addStretch(1)
        self._matrix_row_labels_checkbox = QCheckBox("Show subject/channel row labels")
        self._matrix_row_labels_checkbox.setToolTip(
            "Display one label per matrix row to identify the contributing subject and channel."
        )
        matrix_controls_layout.addWidget(self._matrix_row_labels_checkbox)
        matrix_controls_layout.addStretch(1)
        matrix_layout.addLayout(matrix_controls_layout)
        self._plot_tabs.addTab(matrix_w, "Slope matrix")

        scatter_w = QWidget()
        scatter_layout = QVBoxLayout(scatter_w)
        scatter_layout.setContentsMargins(0, 0, 0, 0)
        self._fig_scatter = Figure(tight_layout=True)
        self._ax_scatter = self._fig_scatter.add_subplot(111)
        self._canvas_scatter = FigureCanvasQTAgg(self._fig_scatter)
        scatter_layout.addWidget(self._canvas_scatter)
        self._plot_tabs.addTab(scatter_w, "Scatter")

        right_layout.addWidget(self._plot_tabs)
        splitter.addWidget(right)
        splitter.setSizes([160, 840])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._current_result: object | None = None

        self._roi_list.currentRowChanged.connect(self._on_roi_changed)
        self._activity_matrix_row_labels_checkbox.toggled.connect(
            self._on_activity_matrix_row_labels_toggled
        )
        self._matrix_row_labels_checkbox.toggled.connect(self._on_matrix_row_labels_toggled)

        self._draw_placeholder("Run group compute to see results")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_plots(
        self,
        result: "ConditionTestGroupProcessingResult | RegressionGroupProcessingResult",
        roi_idx: int,
    ) -> None:
        """Redraw all plots for *roi_idx*."""
        self._current_result = result
        if _is_slope_group_result(result):
            # Show all 9 tabs for slope results
            for i in range(self._plot_tabs.count()):
                self._plot_tabs.setTabVisible(i, True)
            metric_label = _slope_source_metric_label(result)
            metric_tab_label = "Slope" if metric_label == "slope" else metric_label
            self._plot_tabs.setTabText(4, f"{metric_tab_label} mean")
            self._plot_tabs.setTabText(5, f"{metric_tab_label} t-values")
            self._plot_tabs.setTabText(6, f"{metric_tab_label} p-values")
            self._plot_tabs.setTabText(7, f"{metric_tab_label} matrix")
        else:
            # For classic ttest results: hide the 4 activity-specific tabs (indices 1-3)
            # and the slope mean tab (index 4); show activity mean (0), t (5→1), p (6→2), matrix (7→3)
            self._plot_tabs.setTabVisible(1, False)   # Activity t-values
            self._plot_tabs.setTabVisible(2, False)   # Activity p-values
            self._plot_tabs.setTabVisible(3, False)   # Activity matrix
            self._plot_tabs.setTabVisible(4, False)   # Slope mean
            self._plot_tabs.setTabVisible(5, True)
            self._plot_tabs.setTabVisible(6, True)
            self._plot_tabs.setTabVisible(7, True)
            self._plot_tabs.setTabVisible(8, False)   # Scatter (slope only)
            self._plot_tabs.setTabText(0, "Activity")
            self._plot_tabs.setTabText(5, "T-values")
            self._plot_tabs.setTabText(6, "P-values")
            self._plot_tabs.setTabText(7, "Channel Matrix")
            if self._plot_tabs.currentIndex() in (1, 2, 3, 4, 8):
                self._plot_tabs.setCurrentIndex(0)

        self._roi_list.blockSignals(True)
        self._roi_list.clear()
        for name in result.region_names:
            self._roi_list.addItem(QListWidgetItem(name))
        self._roi_list.blockSignals(False)

        if self._roi_list.count() > 0:
            clamped = max(0, min(roi_idx, self._roi_list.count() - 1))
            self._roi_list.setCurrentRow(clamped)

        self._draw_roi(result, roi_idx)

    def show_placeholder(self) -> None:
        """Clear all plots and display a waiting message."""
        self._draw_placeholder("Computing…")

    @property
    def current_roi_index(self) -> int:
        return max(self._roi_list.currentRow(), 0)

    @property
    def current_roi_name(self) -> str | None:
        item = self._roi_list.currentItem()
        return item.text() if item is not None else None

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_roi_changed(self, row: int) -> None:
        if row >= 0 and self._current_result is not None:
            self._draw_roi(self._current_result, row)
        self.roi_changed.emit(row)

    def _on_activity_matrix_row_labels_toggled(self, checked: bool) -> None:
        del checked
        if self._current_result is not None:
            self._draw_roi(self._current_result, self.current_roi_index)

    def _on_matrix_row_labels_toggled(self, checked: bool) -> None:
        del checked
        if self._current_result is not None:
            self._draw_roi(self._current_result, self.current_roi_index)

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    def _draw_roi(self, result: object, roi_idx: int) -> None:
        region_names = getattr(result, "region_names", [])
        if roi_idx < 0 or roi_idx >= len(region_names):
            return
        if _is_slope_group_result(result):
            self._draw_roi_slope(result, roi_idx)
            return
        self._draw_roi_ttest(result, roi_idx)

    # ------------------------------------------------------------------
    # TrialStatsGroup drawing
    # ------------------------------------------------------------------

    def _draw_roi_ttest(
        self,
        result: "ConditionTestGroupProcessingResult",
        roi_idx: int,
    ) -> None:
        roi_label = result.region_names[roi_idx]
        t = result.time_axis_s
        alpha = result.significance_alpha
        sig = (
            result.significant_mask[roi_idx].astype(bool)
            if result.significant_mask.size > 0
            else np.zeros(len(t), dtype=bool)
        )

        ax = self._ax_means
        ax.clear()
        has_cond_data = (
            result.condition_a_group_mean.size > 0
            and result.condition_b_group_mean.size > 0
        )
        if has_cond_data:
            cond_a_label = result.condition_labels[0]
            cond_b_label = result.condition_labels[1]
            mean_a = result.condition_a_group_mean[roi_idx]
            sem_a = result.condition_a_group_sem[roi_idx]
            mean_b = result.condition_b_group_mean[roi_idx]
            sem_b = result.condition_b_group_sem[roi_idx]
            ax.plot(t, mean_a, color="steelblue", label=cond_a_label)
            ax.fill_between(t, mean_a - sem_a, mean_a + sem_a, alpha=0.25, color="steelblue")
            ax.plot(t, mean_b, color="tomato", label=cond_b_label)
            ax.fill_between(t, mean_b - sem_b, mean_b + sem_b, alpha=0.25, color="tomato")
        else:
            ax.text(
                0.5,
                0.5,
                "No condition means available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="gray",
            )
        if has_cond_data:
            all_means = np.concatenate([mean_a, mean_b])
            if np.nanmin(all_means) < 0 < np.nanmax(all_means):
                ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel(_group_activity_axis_label(result))
        n_ch = (
            int(result.roi_channel_counts[roi_idx])
            if result.roi_channel_counts.size > roi_idx
            else "?"
        )
        n_subj = (
            int(result.roi_subject_counts[roi_idx])
            if result.roi_subject_counts.size > roi_idx
            else "?"
        )
        ax.set_title(
            f"{roi_label}  —  {n_ch} channel(s) / {n_subj} subject(s)",
            fontsize=9,
        )
        _safe_legend(ax)
        if sig.any():
            ax.fill_between(
                t,
                0.005,
                0.025,
                where=sig,
                alpha=0.75,
                color="red",
                transform=ax.get_xaxis_transform(),
                zorder=5,
            )
        self._canvas_means.draw_idle()

        ax = self._ax_t
        ax.clear()
        if result.t_values.size > 0:
            t_vals = result.t_values[roi_idx]
            ax.plot(t, t_vals, color="darkorange")
            if np.nanmin(t_vals) < 0 < np.nanmax(t_vals):
                ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        if sig.any():
            ax.fill_between(
                t,
                0,
                1,
                where=sig,
                alpha=0.18,
                color="red",
                transform=ax.get_xaxis_transform(),
            )
        ax.set_ylabel("t-value")
        ax.set_title(
            f"{roi_label} - t-values - {n_ch} channel(s) / {n_subj} subject(s)",
            fontsize=9,
        )
        self._canvas_t.draw_idle()

        ax = self._ax_p
        ax.clear()
        if result.p_values.size > 0:
            p_corr = result.p_values[roi_idx]
            method = result.p_value_correction_method
            has_correction = bool(method) and method.lower() not in ("none", "")
            p_unc = (
                result.p_values_uncorrected[roi_idx]
                if result.p_values_uncorrected.size > 0 and has_correction
                else None
            )
            if p_unc is not None:
                ax.plot(
                    t,
                    p_unc,
                    color="mediumpurple",
                    linewidth=0.9,
                    linestyle="--",
                    alpha=0.7,
                    label="p (uncorrected)",
                )
            corr_label = f"p ({method})" if has_correction else "p-value"
            ax.plot(t, p_corr, color="purple", label=corr_label)
            ax.axhline(alpha, color="red", linewidth=0.8, linestyle="--", label=f"α = {alpha}")
            if sig.any():
                ax.fill_between(
                    t,
                    0,
                    1,
                    where=sig,
                    alpha=0.18,
                    color="red",
                    transform=ax.get_xaxis_transform(),
                )
            all_p = [p_corr]
            if p_unc is not None:
                all_p.append(p_unc)
            p_max = float(np.nanmax(np.concatenate(all_p))) if all_p else 1.0
            ax.set_ylim(bottom=0.0, top=max(1.05, p_max * 1.05))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("p-value")
        ax.set_title(
            f"{roi_label} - p-values - {n_ch} channel(s) / {n_subj} subject(s)",
            fontsize=9,
        )
        _safe_legend(ax)
        self._canvas_p.draw_idle()

        has_contrib = bool(result.condition_a_contributions) and roi_idx < len(result.condition_a_contributions)
        rows_a = (
            result.condition_a_contributions[roi_idx]
            if has_contrib
            else np.empty((0, len(t)), dtype=np.float64)
        )
        rows_b = (
            result.condition_b_contributions[roi_idx]
            if has_contrib
            else np.empty((0, len(t)), dtype=np.float64)
        )
        labels = result.contribution_labels[roi_idx] if has_contrib else []
        self._ax_matrix = self._draw_matrix_on(
            fig=self._fig_matrix,
            canvas=self._canvas_matrix,
            rows_a=rows_a,
            rows_b=rows_b,
            labels=labels,
            show_row_labels=self._matrix_row_labels_checkbox.isChecked(),
            time_axis=t,
            roi_label=roi_label,
            cond_a_label=result.condition_labels[0],
            cond_b_label=result.condition_labels[1],
            title_suffix="",
        )

    # ------------------------------------------------------------------
    # TrialSlopeStatsGroup drawing
    # ------------------------------------------------------------------

    def _draw_roi_slope(
        self,
        result: "RegressionGroupProcessingResult",
        roi_idx: int,
    ) -> None:
        roi_label = result.region_names[roi_idx]
        t = result.time_axis_s
        alpha = result.significance_alpha
        cond_a_label = result.condition_labels[0]
        cond_b_label = result.condition_labels[1]

        sig_slope = (
            result.slope_significant_mask[roi_idx].astype(bool)
            if result.slope_significant_mask.size > 0
            else np.zeros(len(t), dtype=bool)
        )
        sig_activity = (
            result.activity_significant_mask[roi_idx].astype(bool)
            if result.activity_significant_mask.size > 0
            else np.zeros(len(t), dtype=bool)
        )

        n_ch = (
            int(result.roi_channel_counts[roi_idx])
            if result.roi_channel_counts.size > roi_idx
            else "?"
        )
        n_subj = (
            int(result.roi_subject_counts[roi_idx])
            if result.roi_subject_counts.size > roi_idx
            else "?"
        )
        title_base = f"{roi_label}  —  {n_ch} channel(s) / {n_subj} subject(s)"
        method = result.p_value_correction_method
        has_correction = bool(method) and method.lower() not in ("none", "")
        metric_label = _slope_source_metric_label(result)
        contrast_label = _slope_contrast_label(result)

        # ---- Activity means ± SEM ----
        ax = self._ax_means
        ax.clear()
        has_activity = (
            result.condition_a_activity_mean.size > 0
            and result.condition_b_activity_mean.size > 0
        )
        if has_activity:
            mean_a = result.condition_a_activity_mean[roi_idx]
            sem_a = result.condition_a_activity_sem[roi_idx]
            mean_b = result.condition_b_activity_mean[roi_idx]
            sem_b = result.condition_b_activity_sem[roi_idx]
            ax.plot(t, mean_a, color="steelblue", label=cond_a_label)
            ax.fill_between(t, mean_a - sem_a, mean_a + sem_a, alpha=0.25, color="steelblue")
            ax.plot(t, mean_b, color="tomato", label=cond_b_label)
            ax.fill_between(t, mean_b - sem_b, mean_b + sem_b, alpha=0.25, color="tomato")
            all_means = np.concatenate([mean_a, mean_b])
            if np.nanmin(all_means) < 0 < np.nanmax(all_means):
                ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        else:
            ax.text(0.5, 0.5, "No activity means available", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="gray")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel(_group_activity_axis_label(result))
        ax.set_title(title_base, fontsize=9)
        if sig_activity.any():
            ax.fill_between(t, 0.005, 0.025, where=sig_activity, alpha=0.75, color="red",
                            transform=ax.get_xaxis_transform(), zorder=5)
        _safe_legend(ax)
        self._canvas_means.draw_idle()

        # ---- Activity t-values ----
        ax = self._ax_activity_t
        ax.clear()
        if result.activity_t_values.size > 0:
            t_act = result.activity_t_values[roi_idx]
            ax.plot(t, t_act, color="darkorchid",
                    label=f"{cond_a_label} vs {cond_b_label}")
            if np.nanmin(t_act) < 0 < np.nanmax(t_act):
                ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        if sig_activity.any():
            ax.fill_between(t, 0, 1, where=sig_activity, alpha=0.18, color="red",
                            transform=ax.get_xaxis_transform())
        ax.set_ylabel("activity t-value")
        ax.set_title(f"{roi_label} — activity ({cond_a_label} vs {cond_b_label})", fontsize=9)
        _safe_legend(ax)
        self._canvas_activity_t.draw_idle()

        # ---- Activity p-values ----
        ax = self._ax_activity_p
        ax.clear()
        plotted_p_act: list[np.ndarray] = []
        if result.activity_p_values.size > 0:
            p_act = result.activity_p_values[roi_idx]
            plotted_p_act.append(p_act)
            corr_label = f"p ({method})" if has_correction else "p-value"
            ax.plot(t, p_act, color="darkorchid", label=corr_label)
            if has_correction and result.activity_p_values_uncorrected.size > 0:
                p_act_unc = result.activity_p_values_uncorrected[roi_idx]
                plotted_p_act.append(p_act_unc)
                ax.plot(t, p_act_unc, color="darkorchid", linewidth=0.9, linestyle="--",
                        alpha=0.6, label="p (uncorr)")
        ax.axhline(alpha, color="red", linewidth=0.8, linestyle="--", label=f"α = {alpha}")
        if sig_activity.any():
            ax.fill_between(t, 0, 1, where=sig_activity, alpha=0.18, color="red",
                            transform=ax.get_xaxis_transform())
        if plotted_p_act:
            p_max = float(np.nanmax(np.concatenate(plotted_p_act)))
            ax.set_ylim(bottom=0.0, top=max(1.05, p_max * 1.05))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("activity p-value")
        ax.set_title(f"{title_base} - activity p-values", fontsize=9)
        _safe_legend(ax)
        self._canvas_activity_p.draw_idle()

        # ---- Activity matrix ----
        has_act_contrib = bool(result.condition_a_activity_contributions) and roi_idx < len(
            result.condition_a_activity_contributions
        )
        rows_act_a = (
            result.condition_a_activity_contributions[roi_idx]
            if has_act_contrib
            else np.empty((0, len(t)), dtype=np.float64)
        )
        rows_act_b = (
            result.condition_b_activity_contributions[roi_idx]
            if has_act_contrib
            else np.empty((0, len(t)), dtype=np.float64)
        )
        contrib_labels = result.contribution_labels[roi_idx] if has_act_contrib else []
        self._ax_activity_matrix = self._draw_matrix_on(
            fig=self._fig_activity_matrix,
            canvas=self._canvas_activity_matrix,
            rows_a=rows_act_a,
            rows_b=rows_act_b,
            labels=contrib_labels,
            show_row_labels=self._activity_matrix_row_labels_checkbox.isChecked(),
            time_axis=t,
            roi_label=roi_label,
            cond_a_label=cond_a_label,
            cond_b_label=cond_b_label,
            title_suffix=" (activity contributions)",
        )

        # ---- Slope means ± SEM ----
        ax = self._ax_slope
        ax.clear()
        has_slope_mean = (
            result.condition_a_slope_mean.size > 0
            and result.condition_b_slope_mean.size > 0
        )
        if has_slope_mean:
            slope_mean_a = result.condition_a_slope_mean[roi_idx]
            slope_sem_a = result.condition_a_slope_sem[roi_idx]
            slope_mean_b = result.condition_b_slope_mean[roi_idx]
            slope_sem_b = result.condition_b_slope_sem[roi_idx]
            ax.plot(t, slope_mean_a, color="steelblue", label=cond_a_label)
            ax.fill_between(t, slope_mean_a - slope_sem_a, slope_mean_a + slope_sem_a,
                            alpha=0.25, color="steelblue")
            ax.plot(t, slope_mean_b, color="tomato", label=cond_b_label)
            ax.fill_between(t, slope_mean_b - slope_sem_b, slope_mean_b + slope_sem_b,
                            alpha=0.25, color="tomato")
            all_slopes = np.concatenate([slope_mean_a, slope_mean_b])
            if np.nanmin(all_slopes) < 0 < np.nanmax(all_slopes):
                ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        else:
            ax.text(0.5, 0.5, f"No {metric_label} means available", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="gray")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel(f"mean {metric_label}")
        ax.set_title(f"{title_base} - {metric_label} mean", fontsize=9)
        if sig_slope.any():
            ax.fill_between(t, 0.005, 0.025, where=sig_slope, alpha=0.75, color="red",
                            transform=ax.get_xaxis_transform(), zorder=5)
        _safe_legend(ax)
        self._canvas_slope.draw_idle()

        # ---- Slope t-values ----
        ax = self._ax_t
        ax.clear()
        if result.slope_t_values.size > 0:
            t_slope = result.slope_t_values[roi_idx]
            ax.plot(t, t_slope, color="darkorange", label=contrast_label)
            if np.nanmin(t_slope) < 0 < np.nanmax(t_slope):
                ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        if sig_slope.any():
            ax.fill_between(t, 0, 1, where=sig_slope, alpha=0.18, color="red",
                            transform=ax.get_xaxis_transform())
        ax.set_ylabel(f"{metric_label} t-value")
        ax.set_title(f"{roi_label} — {metric_label} ({contrast_label})", fontsize=9)
        _safe_legend(ax)
        self._canvas_t.draw_idle()

        # ---- Slope p-values ----
        ax = self._ax_p
        ax.clear()
        plotted_p_slope: list[np.ndarray] = []
        if result.slope_p_values.size > 0:
            p_slope = result.slope_p_values[roi_idx]
            plotted_p_slope.append(p_slope)
            corr_label = f"p ({method})" if has_correction else "p-value"
            ax.plot(t, p_slope, color="darkorange", label=corr_label)
            if has_correction and result.slope_p_values_uncorrected.size > 0:
                p_slope_unc = result.slope_p_values_uncorrected[roi_idx]
                plotted_p_slope.append(p_slope_unc)
                ax.plot(t, p_slope_unc, color="darkorange", linewidth=0.9, linestyle="--",
                        alpha=0.6, label="p (uncorr)")
        ax.axhline(alpha, color="red", linewidth=0.8, linestyle="--", label=f"α = {alpha}")
        if sig_slope.any():
            ax.fill_between(t, 0, 1, where=sig_slope, alpha=0.18, color="red",
                            transform=ax.get_xaxis_transform())
        if plotted_p_slope:
            p_max = float(np.nanmax(np.concatenate(plotted_p_slope)))
            ax.set_ylim(bottom=0.0, top=max(1.05, p_max * 1.05))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(f"{metric_label} p-value")
        ax.set_title(f"{title_base} - {metric_label} p-values", fontsize=9)
        _safe_legend(ax)
        self._canvas_p.draw_idle()

        # ---- Slope matrix ----
        has_slope_contrib = bool(result.condition_a_slope_contributions) and roi_idx < len(
            result.condition_a_slope_contributions
        )
        rows_slope_a = (
            result.condition_a_slope_contributions[roi_idx]
            if has_slope_contrib
            else np.empty((0, len(t)), dtype=np.float64)
        )
        rows_slope_b = (
            result.condition_b_slope_contributions[roi_idx]
            if has_slope_contrib
            else np.empty((0, len(t)), dtype=np.float64)
        )
        slope_labels = result.contribution_labels[roi_idx] if has_slope_contrib else []
        self._ax_matrix = self._draw_matrix_on(
            fig=self._fig_matrix,
            canvas=self._canvas_matrix,
            rows_a=rows_slope_a,
            rows_b=rows_slope_b,
            labels=slope_labels,
            show_row_labels=self._matrix_row_labels_checkbox.isChecked(),
            time_axis=t,
            roi_label=roi_label,
            cond_a_label=cond_a_label,
            cond_b_label=cond_b_label,
            title_suffix=f" ({metric_label} contributions)",
        )

        # ---- Scatter (predictor vs epoch-mean brain activity) ----
        has_scatter_a = (
            bool(result.condition_a_scatter_predictor)
            and roi_idx < len(result.condition_a_scatter_predictor)
        )
        has_scatter_b = (
            bool(result.condition_b_scatter_predictor)
            and roi_idx < len(result.condition_b_scatter_predictor)
        )
        pred_a = result.condition_a_scatter_predictor[roi_idx] if has_scatter_a else np.empty(0)
        act_a = result.condition_a_scatter_activity[roi_idx] if has_scatter_a else np.empty(0)
        pred_b = result.condition_b_scatter_predictor[roi_idx] if has_scatter_b else np.empty(0)
        act_b = result.condition_b_scatter_activity[roi_idx] if has_scatter_b else np.empty(0)
        self._draw_scatter_on(
            fig=self._fig_scatter,
            canvas=self._canvas_scatter,
            pred_a=pred_a,
            act_a=act_a,
            pred_b=pred_b,
            act_b=act_b,
            roi_label=roi_label,
            cond_a_label=cond_a_label,
            cond_b_label=cond_b_label,
            predictor_label=_group_predictor_axis_label(result),
            activity_summary_label=_group_trial_activity_summary_label(result),
            activity_zscore=_group_activity_zscore(result),
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _draw_matrix_on(
        self,
        *,
        fig: "Figure",
        canvas: "FigureCanvasQTAgg",
        rows_a: np.ndarray,
        rows_b: np.ndarray,
        labels: list[str],
        show_row_labels: bool,
        time_axis: np.ndarray,
        roi_label: str,
        cond_a_label: str,
        cond_b_label: str,
        title_suffix: str,
    ):
        set_layout_engine = getattr(fig, "set_layout_engine", None)
        if callable(set_layout_engine):
            set_layout_engine(None)
        fig.clear()
        ax = fig.add_subplot(111)

        if rows_a.size == 0 and rows_b.size == 0:
            ax.text(
                0.5,
                0.5,
                "No contribution data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="gray",
                fontsize=10,
            )
            ax.set_title(f"{roi_label}{title_suffix}", fontsize=9)
            ax.set_xlabel("Time (s)")
            canvas.draw_idle()
            return ax

        n_t = len(time_axis)
        rows_a = np.asarray(rows_a, dtype=np.float64).reshape(-1, n_t)
        rows_b = np.asarray(rows_b, dtype=np.float64).reshape(-1, n_t)
        n_a = rows_a.shape[0]
        n_b = rows_b.shape[0]
        matrix = np.concatenate([rows_a, rows_b], axis=0)

        vcenter = float(np.nanmean(matrix))
        vrange = float(np.nanpercentile(np.abs(matrix - vcenter), 99)) or 1.0
        im = ax.imshow(
            matrix,
            aspect="auto",
            origin="upper",
            cmap="jet",
            vmin=vcenter - vrange,
            vmax=vcenter + vrange,
            extent=[time_axis[0], time_axis[-1], n_a + n_b - 0.5, -0.5],
            interpolation="nearest",
        )
        fig.colorbar(im, ax=ax, location="right", shrink=0.8)

        if n_a > 0 and n_b > 0:
            ax.axhline(n_a - 0.5, color="white", linewidth=1.5)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")

        if show_row_labels:
            row_labels = _build_matrix_row_labels(
                labels,
                n_a=n_a,
                n_b=n_b,
                cond_a_label=cond_a_label,
                cond_b_label=cond_b_label,
            )
            ax.set_yticks(np.arange(matrix.shape[0], dtype=np.float64))
            ax.set_yticklabels(
                row_labels,
                fontsize=_matrix_row_label_fontsize(matrix.shape[0]),
            )
            ax.tick_params(axis="y", length=0)
        else:
            y_ticks: list[float] = []
            y_tick_labels: list[str] = []
            if n_a > 0:
                y_ticks.append(n_a / 2 - 0.5)
                y_tick_labels.append(cond_a_label)
            if n_b > 0:
                y_ticks.append(n_a + n_b / 2 - 0.5)
                y_tick_labels.append(cond_b_label)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_tick_labels, fontsize=8)

        fig.subplots_adjust(
            left=_matrix_left_margin(show_row_labels, ax.get_yticklabels()),
            right=0.92,
            bottom=0.12,
            top=0.92,
        )

        ax.set_title(f"{roi_label}{title_suffix}", fontsize=9)
        ax.set_xlabel("Time (s)")
        canvas.draw_idle()
        return ax

    def _draw_scatter_on(
        self,
        *,
        fig: "Figure",
        canvas: "FigureCanvasQTAgg",
        pred_a: np.ndarray,
        act_a: np.ndarray,
        pred_b: np.ndarray,
        act_b: np.ndarray,
        roi_label: str,
        cond_a_label: str,
        cond_b_label: str,
        predictor_label: str,
        activity_summary_label: str,
        activity_zscore: str,
    ) -> None:
        """Draw a predictor-vs-activity scatter plot with per-condition summaries."""
        fig.clear()
        self._ax_scatter = fig.add_subplot(111)
        ax = self._ax_scatter

        pred_a = np.asarray(pred_a, dtype=np.float64).ravel()
        act_a = np.asarray(act_a, dtype=np.float64).ravel()
        pred_b = np.asarray(pred_b, dtype=np.float64).ravel()
        act_b = np.asarray(act_b, dtype=np.float64).ravel()

        has_data = pred_a.size > 0 or pred_b.size > 0
        if not has_data:
            ax.text(0.5, 0.5, "No scatter data available", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="gray")
            ax.set_title(f"{roi_label} — scatter", fontsize=9)
            ax.set_xlabel(predictor_label)
            ax.set_ylabel(
                _scatter_activity_axis_label(activity_summary_label, activity_zscore)
            )
            canvas.draw_idle()
            return

        plotted = False
        plotted |= _plot_group_scatter_condition(
            ax=ax,
            predictor_values=pred_a,
            activity_values=act_a,
            color="steelblue",
            label=cond_a_label,
        )
        plotted |= _plot_group_scatter_condition(
            ax=ax,
            predictor_values=pred_b,
            activity_values=act_b,
            color="tomato",
            label=cond_b_label,
        )
        if not plotted:
            ax.text(
                0.5,
                0.5,
                "No scatter data available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="gray",
            )
            ax.set_title(f"{roi_label} — scatter", fontsize=9)
            ax.set_xlabel(predictor_label)
            ax.set_ylabel(
                _scatter_activity_axis_label(activity_summary_label, activity_zscore)
            )
            canvas.draw_idle()
            return

        ax.set_xlabel(predictor_label)
        ax.set_ylabel(_scatter_activity_axis_label(activity_summary_label, activity_zscore))
        ax.set_title(f"{roi_label} — predictor vs activity", fontsize=9)
        _safe_legend(ax)
        canvas.draw_idle()

    def _draw_placeholder(self, message: str = "") -> None:
        for ax in [self._ax_means, self._ax_activity_t, self._ax_activity_p, self._ax_slope, self._ax_t, self._ax_p, self._ax_scatter]:
            ax.clear()
            ax.set_facecolor("#f4f4f4")
            ax.set_xticks([])
            ax.set_yticks([])
        for fig, ax_attr in [
            (self._fig_activity_matrix, "_ax_activity_matrix"),
            (self._fig_matrix, "_ax_matrix"),
            (self._fig_scatter, "_ax_scatter"),
        ]:
            fig.clear()
            new_ax = fig.add_subplot(111)
            setattr(self, ax_attr, new_ax)
            new_ax.set_facecolor("#f4f4f4")
            new_ax.set_xticks([])
            new_ax.set_yticks([])
        if message:
            self._ax_means.text(
                0.5,
                0.5,
                message,
                transform=self._ax_means.transAxes,
                ha="center",
                va="center",
                fontsize=10,
                color="gray",
            )
        for canvas in [
            self._canvas_means,
            self._canvas_activity_t,
            self._canvas_activity_p,
            self._canvas_activity_matrix,
            self._canvas_slope,
            self._canvas_t,
            self._canvas_p,
            self._canvas_matrix,
            self._canvas_scatter,
        ]:
            canvas.draw_idle()


def _safe_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(fontsize="small", loc="upper right")


def _build_matrix_row_labels(
    labels: list[str],
    *,
    n_a: int,
    n_b: int,
    cond_a_label: str,
    cond_b_label: str,
) -> list[str]:
    raw_labels = list(labels)
    if len(raw_labels) == n_a + n_b:
        labels_a = raw_labels[:n_a]
        labels_b = raw_labels[n_a:]
    else:
        labels_a = raw_labels[:n_a]
        labels_b = raw_labels[:n_b]

    return _build_matrix_block_row_labels(labels_a, n_a, cond_a_label) + _build_matrix_block_row_labels(
        labels_b,
        n_b,
        cond_b_label,
    )


def _build_matrix_block_row_labels(
    labels: list[str],
    n_rows: int,
    block_label: str,
) -> list[str]:
    out: list[str] = []
    for idx in range(n_rows):
        base_label = _format_contribution_label(labels[idx]) if idx < len(labels) else f"row {idx + 1}"
        out.append(f"{block_label}: {base_label}")
    return out


def _format_contribution_label(label: str) -> str:
    raw_label = str(label).strip()
    if not raw_label:
        return "unknown"
    if "/" not in raw_label:
        return raw_label
    subject, channel = raw_label.split("/", 1)
    subject = subject.strip()
    channel = channel.strip()
    if subject and not subject.startswith("sub-"):
        subject = f"sub-{subject}"
    if channel:
        return f"{subject} / {channel}"
    return subject or raw_label


def _matrix_row_label_fontsize(n_rows: int) -> int:
    if n_rows <= 12:
        return 8
    if n_rows <= 24:
        return 7
    if n_rows <= 40:
        return 6
    return 5


def _matrix_left_margin(show_row_labels: bool, tick_labels: list) -> float:
    if not show_row_labels:
        return 0.12
    label_lengths = [len(label.get_text()) for label in tick_labels if label.get_text()]
    if not label_lengths:
        return 0.18
    return min(0.48, max(0.18, 0.011 * max(label_lengths)))


def _is_slope_group_result(result: object) -> bool:
    return hasattr(result, "slope_t_values")


def _group_activity_axis_label(result: object) -> str:
    if _is_group_activity_zscore_enabled(_group_activity_zscore(result)):
        return "mean region activity (z)"
    return "mean region activity"


def _group_trial_activity_summary_label(result: object) -> str:
    metadata = getattr(result, "metadata", {})
    if isinstance(metadata, dict):
        label = str(metadata.get("trial_activity_summary_label", "")).strip()
        if label:
            return label
    return "Epoch mean activity"


def _group_activity_zscore(result: object) -> str:
    metadata = getattr(result, "metadata", {})
    if isinstance(metadata, dict):
        return str(metadata.get("activity_zscore", "none"))
    return "none"


def _is_group_activity_zscore_enabled(activity_zscore: str) -> bool:
    return str(activity_zscore).strip().lower() in {"baseline", "across_trials"}


def _plot_group_scatter_condition(
    *,
    ax,
    predictor_values: np.ndarray,
    activity_values: np.ndarray,
    color: str,
    label: str,
) -> bool:
    predictor = np.asarray(predictor_values, dtype=np.float64).ravel()
    activity = np.asarray(activity_values, dtype=np.float64).ravel()
    valid = np.isfinite(predictor) & np.isfinite(activity)
    if valid.sum() == 0:
        return False

    predictor = predictor[valid]
    activity = activity[valid]
    ax.scatter(
        predictor,
        activity,
        color=color,
        alpha=0.35,
        s=18,
        label=label,
        linewidths=0,
        zorder=1,
    )
    if predictor.size >= 2:
        regression = _fit_scatter_regression(predictor, activity)
        if regression is not None:
            x_range = np.array([predictor.min(), predictor.max()], dtype=np.float64)
            y_fit = regression.intercept + (regression.slope * x_range)
            ax.plot(
                x_range,
                y_fit,
                color=color,
                linewidth=1.8 if regression.is_significant else 1.2,
                linestyle="-" if regression.is_significant else "--",
                zorder=2,
            )
    summary_x, summary_y, summary_x_sem, summary_y_sem = _compute_scatter_summary_points(
        predictor,
        activity,
        target_bins=_SCATTER_SUMMARY_TARGET_BINS,
    )
    if summary_x.size > 0:
        ax.errorbar(
            summary_x,
            summary_y,
            xerr=summary_x_sem,
            yerr=summary_y_sem,
            fmt="o",
            linestyle="none",
            color=color,
            ecolor=color,
            elinewidth=1.6,
            capsize=0,
            markersize=8,
            markerfacecolor=color,
            markeredgecolor="black",
            markeredgewidth=1.0,
            alpha=1.0,
            zorder=3,
            label="_nolegend_",
        )
    return True


def _group_predictor_axis_label(result: object) -> str:
    metadata = getattr(result, "metadata", {})
    predictor = ""
    predictor_zscore = "none"
    if isinstance(metadata, dict):
        predictor = str(metadata.get("predictor", ""))
        predictor_zscore = str(metadata.get("predictor_zscore", "none"))
    base = predictor.strip() or "Predictor value"
    if predictor_zscore.strip().lower() in {"condition", "global"}:
        return f"{base} (z)"
    return base


def _slope_source_metric_label(result: object) -> str:
    metric = str(getattr(result, "source_metric", "slope")).strip().lower()
    if metric == "r_value":
        return "r"
    return "slope"


def _slope_contrast_label(result: object) -> str:
    labels = getattr(result, "condition_labels", ("condition_a", "condition_b"))
    cond_a = labels[0] if len(labels) >= 1 else "condition_a"
    cond_b = labels[1] if len(labels) >= 2 else "condition_b"
    mode = str(getattr(result, "contrast_mode", "paired")).strip().lower()
    return f"{cond_a} vs {cond_b} ({mode})"

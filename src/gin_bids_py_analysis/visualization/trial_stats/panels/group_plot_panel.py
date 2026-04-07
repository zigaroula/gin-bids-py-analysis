"""Group-level plot panel: ROI selector + tabbed matplotlib plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from gin_bids_py_analysis.processing.trial_slope_stats_group.result import (
        TrialSlopeStatsGroupProcessingResult,
    )
    from gin_bids_py_analysis.processing.trial_stats_group.result import (
        TrialStatsGroupProcessingResult,
    )


class GroupPlotPanel(QWidget):
    """Panel combining a ROI selector list and tabbed group plots.

    The panel supports two result schemas:
    - ``TrialStatsGroupProcessingResult`` (classic t-test group stats)
    - ``TrialSlopeStatsGroupProcessingResult`` (slope-regression group stats)
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
        self._plot_tabs.addTab(activity_w, "Activity")

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
        self._plot_tabs.addTab(t_w, "T-values")

        p_w = QWidget()
        p_layout = QVBoxLayout(p_w)
        p_layout.setContentsMargins(0, 0, 0, 0)
        self._fig_p = Figure(tight_layout=True)
        self._ax_p = self._fig_p.add_subplot(111)
        self._canvas_p = FigureCanvasQTAgg(self._fig_p)
        p_layout.addWidget(self._canvas_p)
        self._plot_tabs.addTab(p_w, "P-values")

        matrix_w = QWidget()
        matrix_layout = QVBoxLayout(matrix_w)
        matrix_layout.setContentsMargins(0, 0, 0, 0)
        self._fig_matrix = Figure(tight_layout=True)
        self._ax_matrix = self._fig_matrix.add_subplot(111)
        self._canvas_matrix = FigureCanvasQTAgg(self._fig_matrix)
        matrix_layout.addWidget(self._canvas_matrix)
        self._plot_tabs.addTab(matrix_w, "Channel Matrix")

        right_layout.addWidget(self._plot_tabs)
        splitter.addWidget(right)
        splitter.setSizes([160, 840])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._current_result: object | None = None

        self._roi_list.currentRowChanged.connect(self._on_roi_changed)

        self._draw_placeholder("Run group compute to see results")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_plots(
        self,
        result: "TrialStatsGroupProcessingResult | TrialSlopeStatsGroupProcessingResult",
        roi_idx: int,
    ) -> None:
        """Redraw all plots for *roi_idx*."""
        self._current_result = result
        if _is_slope_group_result(result):
            self._plot_tabs.setTabVisible(1, True)
            self._plot_tabs.setTabText(0, "Activity")
            self._plot_tabs.setTabText(1, "Slope mean")
            self._plot_tabs.setTabText(2, "Slope t-values")
            self._plot_tabs.setTabText(3, "Slope p-values")
            self._plot_tabs.setTabText(4, "Slope Matrix")
        else:
            self._plot_tabs.setTabVisible(1, False)
            if self._plot_tabs.currentIndex() == 1:
                self._plot_tabs.setCurrentIndex(0)
            self._plot_tabs.setTabText(0, "Activity")
            self._plot_tabs.setTabText(2, "T-values")
            self._plot_tabs.setTabText(3, "P-values")
            self._plot_tabs.setTabText(4, "Channel Matrix")

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
        result: "TrialStatsGroupProcessingResult",
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
        ax.set_ylabel("mean region activity")
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
        self._draw_matrix(
            rows_a=rows_a,
            rows_b=rows_b,
            labels=labels,
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
        result: "TrialSlopeStatsGroupProcessingResult",
        roi_idx: int,
    ) -> None:
        roi_label = result.region_names[roi_idx]
        t = result.time_axis_s
        alpha = result.significance_alpha
        cond_a_label = result.condition_labels[0]
        cond_b_label = result.condition_labels[1]

        sig_a = (
            result.condition_a_slope_significant_mask[roi_idx].astype(bool)
            if result.condition_a_slope_significant_mask.size > 0
            else np.zeros(len(t), dtype=bool)
        )
        sig_b = (
            result.condition_b_slope_significant_mask[roi_idx].astype(bool)
            if result.condition_b_slope_significant_mask.size > 0
            else np.zeros(len(t), dtype=bool)
        )
        sig_any = sig_a | sig_b

        # Activity means ± SEM
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
            ax.text(
                0.5,
                0.5,
                "No activity means available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="gray",
            )
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel("mean region activity")
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
        if sig_a.any():
            ax.fill_between(
                t,
                0.005,
                0.025,
                where=sig_a,
                alpha=0.75,
                color="red",
                transform=ax.get_xaxis_transform(),
                zorder=5,
            )
        if sig_b.any():
            ax.fill_between(
                t,
                0.03,
                0.05,
                where=sig_b,
                alpha=0.75,
                color="orange",
                transform=ax.get_xaxis_transform(),
                zorder=5,
            )
        self._canvas_means.draw_idle()

        # Slope means ± SEM
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
            ax.fill_between(
                t,
                slope_mean_a - slope_sem_a,
                slope_mean_a + slope_sem_a,
                alpha=0.25,
                color="steelblue",
            )
            ax.plot(t, slope_mean_b, color="tomato", label=cond_b_label)
            ax.fill_between(
                t,
                slope_mean_b - slope_sem_b,
                slope_mean_b + slope_sem_b,
                alpha=0.25,
                color="tomato",
            )
            all_slopes = np.concatenate([slope_mean_a, slope_mean_b])
            if np.nanmin(all_slopes) < 0 < np.nanmax(all_slopes):
                ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        else:
            ax.text(
                0.5,
                0.5,
                "No slope means available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color="gray",
            )
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel("mean slope")
        if sig_a.any():
            ax.fill_between(
                t,
                0.005,
                0.025,
                where=sig_a,
                alpha=0.75,
                color="red",
                transform=ax.get_xaxis_transform(),
                zorder=5,
            )
        if sig_b.any():
            ax.fill_between(
                t,
                0.03,
                0.05,
                where=sig_b,
                alpha=0.75,
                color="orange",
                transform=ax.get_xaxis_transform(),
                zorder=5,
            )
        _safe_legend(ax)
        self._canvas_slope.draw_idle()

        ax = self._ax_t
        ax.clear()
        if result.condition_a_slope_t_values.size > 0:
            t_a = result.condition_a_slope_t_values[roi_idx]
            ax.plot(t, t_a, color="steelblue", label=f"t slope {cond_a_label}")
        else:
            t_a = np.empty(0, dtype=np.float64)
        if result.condition_b_slope_t_values.size > 0:
            t_b = result.condition_b_slope_t_values[roi_idx]
            ax.plot(t, t_b, color="tomato", label=f"t slope {cond_b_label}")
        else:
            t_b = np.empty(0, dtype=np.float64)
        if t_a.size > 0 or t_b.size > 0:
            combined_t = np.concatenate([arr for arr in (t_a, t_b) if arr.size > 0])
            if np.nanmin(combined_t) < 0 < np.nanmax(combined_t):
                ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        if sig_any.any():
            ax.fill_between(
                t,
                0,
                1,
                where=sig_any,
                alpha=0.18,
                color="red",
                transform=ax.get_xaxis_transform(),
            )
        ax.set_ylabel("slope t-value")
        _safe_legend(ax)
        self._canvas_t.draw_idle()

        ax = self._ax_p
        ax.clear()
        method = result.p_value_correction_method
        has_correction = bool(method) and method.lower() not in ("none", "")
        plotted_arrays: list[np.ndarray] = []

        if result.condition_a_slope_p_values.size > 0:
            p_a = result.condition_a_slope_p_values[roi_idx]
            plotted_arrays.append(p_a)
            ax.plot(t, p_a, color="steelblue", label=f"p {cond_a_label}")
            if has_correction and result.condition_a_slope_p_values_uncorrected.size > 0:
                p_a_unc = result.condition_a_slope_p_values_uncorrected[roi_idx]
                plotted_arrays.append(p_a_unc)
                ax.plot(
                    t,
                    p_a_unc,
                    color="steelblue",
                    linewidth=0.9,
                    linestyle="--",
                    alpha=0.6,
                    label=f"p {cond_a_label} (uncorr)",
                )
        if result.condition_b_slope_p_values.size > 0:
            p_b = result.condition_b_slope_p_values[roi_idx]
            plotted_arrays.append(p_b)
            ax.plot(t, p_b, color="tomato", label=f"p {cond_b_label}")
            if has_correction and result.condition_b_slope_p_values_uncorrected.size > 0:
                p_b_unc = result.condition_b_slope_p_values_uncorrected[roi_idx]
                plotted_arrays.append(p_b_unc)
                ax.plot(
                    t,
                    p_b_unc,
                    color="tomato",
                    linewidth=0.9,
                    linestyle="--",
                    alpha=0.6,
                    label=f"p {cond_b_label} (uncorr)",
                )

        ax.axhline(alpha, color="red", linewidth=0.8, linestyle="--", label=f"α = {alpha}")
        if sig_any.any():
            ax.fill_between(
                t,
                0,
                1,
                where=sig_any,
                alpha=0.18,
                color="red",
                transform=ax.get_xaxis_transform(),
            )
        if plotted_arrays:
            p_max = float(np.nanmax(np.concatenate(plotted_arrays)))
            ax.set_ylim(bottom=0.0, top=max(1.05, p_max * 1.05))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("slope p-value")
        _safe_legend(ax)
        self._canvas_p.draw_idle()

        has_contrib = bool(result.condition_a_slope_contributions) and roi_idx < len(
            result.condition_a_slope_contributions
        )
        rows_a = (
            result.condition_a_slope_contributions[roi_idx]
            if has_contrib
            else np.empty((0, len(t)), dtype=np.float64)
        )
        rows_b = (
            result.condition_b_slope_contributions[roi_idx]
            if has_contrib
            else np.empty((0, len(t)), dtype=np.float64)
        )
        labels = result.contribution_labels[roi_idx] if has_contrib else []
        self._draw_matrix(
            rows_a=rows_a,
            rows_b=rows_b,
            labels=labels,
            time_axis=t,
            roi_label=roi_label,
            cond_a_label=f"slope {cond_a_label}",
            cond_b_label=f"slope {cond_b_label}",
            title_suffix=" (slope contributions)",
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _draw_matrix(
        self,
        *,
        rows_a: np.ndarray,
        rows_b: np.ndarray,
        labels: list[str],
        time_axis: np.ndarray,
        roi_label: str,
        cond_a_label: str,
        cond_b_label: str,
        title_suffix: str,
    ) -> None:
        self._fig_matrix.clear()
        self._ax_matrix = self._fig_matrix.add_subplot(111)
        ax = self._ax_matrix

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
            ax.set_xlabel("Time (s)")
            self._canvas_matrix.draw_idle()
            return

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
        self._fig_matrix.colorbar(im, ax=ax, location="right", shrink=0.8)

        if n_a > 0 and n_b > 0:
            ax.axhline(n_a - 0.5, color="white", linewidth=1.5)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")

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

        n_labels = len(labels)
        if n_labels == matrix.shape[0]:
            step = max(1, int(np.ceil(matrix.shape[0] / 24)))
            y_tick_positions = list(range(0, matrix.shape[0], step))
            y_tick_names = [labels[i] for i in y_tick_positions]
            ax.set_yticks(y_tick_positions)
            ax.set_yticklabels(y_tick_names, fontsize=7)

        ax.set_title(f"{roi_label}{title_suffix}", fontsize=9)
        ax.set_xlabel("Time (s)")
        self._canvas_matrix.draw_idle()

    def _draw_placeholder(self, message: str = "") -> None:
        for ax in [self._ax_means, self._ax_slope, self._ax_t, self._ax_p]:
            ax.clear()
            ax.set_facecolor("#f4f4f4")
            ax.set_xticks([])
            ax.set_yticks([])
        self._fig_matrix.clear()
        self._ax_matrix = self._fig_matrix.add_subplot(111)
        self._ax_matrix.set_facecolor("#f4f4f4")
        self._ax_matrix.set_xticks([])
        self._ax_matrix.set_yticks([])
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
            self._canvas_slope,
            self._canvas_t,
            self._canvas_p,
            self._canvas_matrix,
        ]:
            canvas.draw_idle()


def _safe_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(fontsize="small", loc="upper right")


def _is_slope_group_result(result: object) -> bool:
    return hasattr(result, "condition_a_slope_t_values")

"""Group-level plot panel: ROI selector + four synchronized matplotlib plots."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from gin_bids_py_analysis.processing.trial_stats_group.result import (
    TrialStatsGroupProcessingResult,
)


class GroupPlotPanel(QWidget):
    """Panel combining a ROI selector list and four synchronized plots.

    Layout
    ------
    Left: ``QListWidget`` — ROI names.
    Right: Four stacked plots (top → bottom):

    1. Metric mean ± 1 SEM over time.
    2. T-values over time.
    3. P-values + significance shading.
    4. Epoch summary — one bar per ROI showing mean t-value ± SEM over the epoch.

    Signals
    -------
    roi_changed : int  — emitted with the new row index when the ROI selection changes.
    """

    roi_changed = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter)

        # Left: ROI list
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.addWidget(QLabel("ROIs"))
        self._roi_list = QListWidget()
        left_layout.addWidget(self._roi_list, stretch=1)
        splitter.addWidget(left)

        # Right: four plots
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(2)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self._fig_means = Figure(tight_layout=True)
        self._ax_means = self._fig_means.add_subplot(111)
        self._canvas_means = FigureCanvasQTAgg(self._fig_means)
        right_layout.addWidget(self._canvas_means, stretch=1)

        self._fig_t = Figure(tight_layout=True)
        self._ax_t = self._fig_t.add_subplot(111)
        self._canvas_t = FigureCanvasQTAgg(self._fig_t)
        right_layout.addWidget(self._canvas_t, stretch=1)

        self._fig_p = Figure(tight_layout=True)
        self._ax_p = self._fig_p.add_subplot(111)
        self._canvas_p = FigureCanvasQTAgg(self._fig_p)
        right_layout.addWidget(self._canvas_p, stretch=1)

        self._fig_epoch = Figure(tight_layout=True)
        self._ax_epoch = self._fig_epoch.add_subplot(111)
        self._canvas_epoch = FigureCanvasQTAgg(self._fig_epoch)
        right_layout.addWidget(self._canvas_epoch, stretch=1)

        splitter.addWidget(right)
        splitter.setSizes([160, 840])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self._current_result: TrialStatsGroupProcessingResult | None = None

        self._roi_list.currentRowChanged.connect(self._on_roi_changed)

        self._draw_placeholder("Run group compute to see results")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_plots(
        self,
        result: TrialStatsGroupProcessingResult,
        roi_idx: int,
    ) -> None:
        """Redraw the first three plots for *roi_idx* and the epoch summary for all ROIs."""
        self._current_result = result

        # Repopulate ROI list (preserve selection)
        self._roi_list.blockSignals(True)
        self._roi_list.clear()
        for name in result.region_names:
            self._roi_list.addItem(QListWidgetItem(name))
        self._roi_list.blockSignals(False)
        if self._roi_list.count() > 0:
            clamped = max(0, min(roi_idx, self._roi_list.count() - 1))
            self._roi_list.setCurrentRow(clamped)

        self._draw_roi(result, roi_idx)
        self._draw_epoch_summary(result)

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
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _draw_roi(
        self,
        result: TrialStatsGroupProcessingResult,
        roi_idx: int,
    ) -> None:
        if roi_idx < 0 or roi_idx >= len(result.region_names):
            return
        roi_label = result.region_names[roi_idx]
        t = result.time_axis_s
        alpha = result.significance_alpha

        # --- Plot 1: metric mean ± SEM ---
        ax = self._ax_means
        ax.clear()
        if result.metric_mean.size > 0 and result.metric_sem.size > 0:
            mean = result.metric_mean[roi_idx]
            sem = result.metric_sem[roi_idx]
            ax.plot(t, mean, color="steelblue", label=result.source_metric)
            ax.fill_between(t, mean - sem, mean + sem, alpha=0.25, color="steelblue")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel(result.source_metric)
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
        ax.legend(fontsize="small", loc="upper right")
        self._canvas_means.draw_idle()

        # --- Plot 2: t-values ---
        ax = self._ax_t
        ax.clear()
        if result.t_values.size > 0:
            ax.plot(t, result.t_values[roi_idx], color="darkorange")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel("t-value")
        self._canvas_t.draw_idle()

        # --- Plot 3: p-values + significance ---
        ax = self._ax_p
        ax.clear()
        if result.p_values.size > 0:
            p = result.p_values[roi_idx]
            sig = result.significant_mask[roi_idx].astype(bool) if result.significant_mask.size > 0 else np.zeros_like(p, dtype=bool)
            ax.plot(t, p, color="purple")
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
            p_max = float(np.nanmax(p)) if len(p) > 0 else 1.0
            ax.set_ylim(bottom=0.0, top=max(1.05, p_max * 1.05))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("p-value")
        ax.legend(fontsize="small", loc="upper right")
        self._canvas_p.draw_idle()

    def _draw_epoch_summary(
        self,
        result: TrialStatsGroupProcessingResult,
    ) -> None:
        """Bar chart: epoch-mean t-value ± SEM for every ROI."""
        ax = self._ax_epoch
        ax.clear()
        if result.epoch_mean_t_values.size == 0 or not result.region_names:
            ax.set_title("Epoch summary (no data)", fontsize=9)
            self._canvas_epoch.draw_idle()
            return

        n_rois = len(result.region_names)
        x = np.arange(n_rois)
        t_means = result.epoch_mean_t_values
        t_sems = result.epoch_mean_metric_sem if result.epoch_mean_metric_sem.size > 0 else np.zeros(n_rois)

        # Highlight significant ROIs
        sig_mask = (
            (result.epoch_mean_p_values < result.significance_alpha)
            if result.epoch_mean_p_values.size > 0
            else np.zeros(n_rois, dtype=bool)
        )
        colors = ["tomato" if s else "steelblue" for s in sig_mask]

        ax.bar(x, t_means, yerr=t_sems, color=colors, capsize=3, ecolor="black", error_kw={"linewidth": 0.8})
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(result.region_names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("mean t-value")
        ax.set_title("Epoch summary (red = significant)", fontsize=9)
        self._canvas_epoch.draw_idle()

    def _draw_placeholder(self, message: str = "") -> None:
        for ax, canvas in [
            (self._ax_means, self._canvas_means),
            (self._ax_t, self._canvas_t),
            (self._ax_p, self._canvas_p),
            (self._ax_epoch, self._canvas_epoch),
        ]:
            ax.clear()
            ax.set_facecolor("#f4f4f4")
            ax.set_xticks([])
            ax.set_yticks([])
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
            self._canvas_t,
            self._canvas_p,
            self._canvas_epoch,
        ]:
            canvas.draw_idle()

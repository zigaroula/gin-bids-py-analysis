"""Group-level plot panel: ROI selector + three tabbed matplotlib plots."""

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
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt

from gin_bids_py_analysis.processing.trial_stats_group.result import (
    TrialStatsGroupProcessingResult,
)


class GroupPlotPanel(QWidget):
    """Panel combining a ROI selector list and three tabbed plots.

    Layout
    ------
    Left: ``QListWidget`` — ROI names.
    Right: Four tabs:

    * **Activity**      — Condition A and B mean ± SEM over time; significance shown
      as a thin bar at the bottom of the axes.
    * **T-values**      — T-values over time with significance shading.
    * **P-values**      — P-values with significance threshold and shading.
    * **Channel Matrix**  — Heatmap of per-contribution (subject/channel) time series:
      condition A contributions on top, condition B below.

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

        # Right: three tabbed plots
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(0)
        right_layout.setContentsMargins(4, 4, 4, 4)

        self._plot_tabs = QTabWidget()

        activity_w = QWidget()
        al = QVBoxLayout(activity_w)
        al.setContentsMargins(0, 0, 0, 0)
        self._fig_means = Figure(tight_layout=True)
        self._ax_means = self._fig_means.add_subplot(111)
        self._canvas_means = FigureCanvasQTAgg(self._fig_means)
        al.addWidget(self._canvas_means)
        self._plot_tabs.addTab(activity_w, "Activity")

        t_w = QWidget()
        tl = QVBoxLayout(t_w)
        tl.setContentsMargins(0, 0, 0, 0)
        self._fig_t = Figure(tight_layout=True)
        self._ax_t = self._fig_t.add_subplot(111)
        self._canvas_t = FigureCanvasQTAgg(self._fig_t)
        tl.addWidget(self._canvas_t)
        self._plot_tabs.addTab(t_w, "T-values")

        p_w = QWidget()
        pl = QVBoxLayout(p_w)
        pl.setContentsMargins(0, 0, 0, 0)
        self._fig_p = Figure(tight_layout=True)
        self._ax_p = self._fig_p.add_subplot(111)
        self._canvas_p = FigureCanvasQTAgg(self._fig_p)
        pl.addWidget(self._canvas_p)
        self._plot_tabs.addTab(p_w, "P-values")

        matrix_w = QWidget()
        ml = QVBoxLayout(matrix_w)
        ml.setContentsMargins(0, 0, 0, 0)
        self._fig_matrix = Figure(tight_layout=True)
        self._ax_matrix = self._fig_matrix.add_subplot(111)
        self._canvas_matrix = FigureCanvasQTAgg(self._fig_matrix)
        ml.addWidget(self._canvas_matrix)
        self._plot_tabs.addTab(matrix_w, "Channel Matrix")

        right_layout.addWidget(self._plot_tabs)
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
        """Redraw all three plots for *roi_idx*."""
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
        sig = (
            result.significant_mask[roi_idx].astype(bool)
            if result.significant_mask.size > 0
            else np.zeros(len(t), dtype=bool)
        )

        # --- Plot 1: condition A and condition B mean ± SEM ---
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
                0.5, 0.5, "No condition means available",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="gray",
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
        ax.legend(fontsize="small", loc="upper right")
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

        # --- Plot 2: t-values ---
        ax = self._ax_t
        ax.clear()
        if result.t_values.size > 0:
            tv = result.t_values[roi_idx]
            ax.plot(t, tv, color="darkorange")
            if np.nanmin(tv) < 0 < np.nanmax(tv):
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

        # --- Plot 3: p-values + significance ---
        ax = self._ax_p
        ax.clear()
        if result.p_values.size > 0:
            p = result.p_values[roi_idx]
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

        # --- Channel Matrix tab: per-contribution heatmap ---
        self._fig_matrix.clear()
        self._ax_matrix = self._fig_matrix.add_subplot(111)
        ax = self._ax_matrix
        has_contrib = (
            bool(result.condition_a_contributions)
            and roi_idx < len(result.condition_a_contributions)
        )
        if not has_contrib:
            ax.text(0.5, 0.5, "No contribution data", transform=ax.transAxes,
                    ha="center", va="center", color="gray", fontsize=10)
        else:
            rows_a = result.condition_a_contributions[roi_idx]  # (n_contrib, n_times)
            rows_b = result.condition_b_contributions[roi_idx]  # (n_contrib, n_times)
            labels = result.contribution_labels[roi_idx]
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
                extent=[t[0], t[-1], n_a + n_b - 0.5, -0.5],
                interpolation="nearest",
            )
            self._fig_matrix.colorbar(im, ax=ax, location="right", shrink=0.8)
            if n_a > 0 and n_b > 0:
                ax.axhline(n_a - 0.5, color="white", linewidth=1.5)
            ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
            cond_a_label = result.condition_labels[0]
            cond_b_label = result.condition_labels[1]
            y_ticks = []
            y_tick_labels = []
            if n_a > 0:
                y_ticks.append(n_a / 2 - 0.5)
                y_tick_labels.append(cond_a_label)
            if n_b > 0:
                y_ticks.append(n_a + n_b / 2 - 0.5)
                y_tick_labels.append(cond_b_label)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_tick_labels, fontsize=8)
            ax.set_title(
                f"{roi_label}",
                fontsize=9,
            )
        ax.set_xlabel("Time (s)")
        self._canvas_matrix.draw_idle()

    def _draw_placeholder(self, message: str = "") -> None:
        for ax, canvas in [
            (self._ax_means, self._canvas_means),
            (self._ax_t, self._canvas_t),
            (self._ax_p, self._canvas_p),
        ]:
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
            self._canvas_t,
            self._canvas_p,
            self._canvas_matrix,
        ]:
            canvas.draw_idle()

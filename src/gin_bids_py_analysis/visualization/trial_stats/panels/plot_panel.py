"""Middle panel: three stacked matplotlib plots."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QVBoxLayout, QWidget

from gin_bids_py_analysis.processing.trial_stats import TrialStatsProcessingResult


class PlotPanel(QWidget):
    """Middle panel showing three synchronized plots for trial stats results.

    Plots (top to bottom)
    ----------------------
    1. Condition means with ±1 SEM shading.
    2. T-values.
    3. P-values with significance threshold and significant-interval shading.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(2)
        layout.setContentsMargins(4, 4, 4, 4)

        self._fig_means = Figure(tight_layout=True)
        self._ax_means = self._fig_means.add_subplot(111)
        self._canvas_means = FigureCanvasQTAgg(self._fig_means)
        layout.addWidget(self._canvas_means, stretch=1)

        self._fig_t = Figure(tight_layout=True)
        self._ax_t = self._fig_t.add_subplot(111)
        self._canvas_t = FigureCanvasQTAgg(self._fig_t)
        layout.addWidget(self._canvas_t, stretch=1)

        self._fig_p = Figure(tight_layout=True)
        self._ax_p = self._fig_p.add_subplot(111)
        self._canvas_p = FigureCanvasQTAgg(self._fig_p)
        layout.addWidget(self._canvas_p, stretch=1)

        self._draw_placeholder("Select a subject and click Compute")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_plots(
        self,
        result: TrialStatsProcessingResult,
        channel_idx: int,
    ) -> None:
        """Redraw all three plots for the given channel index.

        If ``result.stats_valid`` is False, a notice is shown instead.
        """
        if not result.stats_valid:
            self._draw_placeholder(
                "Not enough trials to compute statistics\n"
                f"({result.condition_a_trial_count}×{result.condition_a}, "
                f"{result.condition_b_trial_count}×{result.condition_b})"
            )
            return

        ch = channel_idx
        t = result.time_axis_s
        ch_label = result.channel_names[ch]

        # --- Plot 1: Means + SEM ---
        ax = self._ax_means
        ax.clear()
        mean_a = result.condition_a_mean[ch]
        mean_b = result.condition_b_mean[ch]
        sem_a = result.condition_a_sem[ch]
        sem_b = result.condition_b_sem[ch]
        ax.plot(t, mean_a, color="steelblue", label=result.condition_a)
        ax.fill_between(t, mean_a - sem_a, mean_a + sem_a, alpha=0.25, color="steelblue")
        ax.plot(t, mean_b, color="tomato", label=result.condition_b)
        ax.fill_between(t, mean_b - sem_b, mean_b + sem_b, alpha=0.25, color="tomato")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Amplitude")
        ax.set_title(
            f"{ch_label}  —  "
            f"{result.condition_a_trial_count}× {result.condition_a} / "
            f"{result.condition_b_trial_count}× {result.condition_b}",
            fontsize=9,
        )
        ax.legend(fontsize="small", loc="upper right")
        self._canvas_means.draw_idle()

        # --- Plot 2: T-values ---
        ax = self._ax_t
        ax.clear()
        ax.plot(t, result.t_values[ch], color="darkorange")
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel("t-value")
        self._canvas_t.draw_idle()

        # --- Plot 3: P-values ---
        ax = self._ax_p
        ax.clear()
        alpha = result.significance_alpha
        p = result.p_values[ch]
        sig = result.significant_mask[ch].astype(bool)
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
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("p-value")
        p_max = float(np.nanmax(p)) if len(p) > 0 else 1.0
        ax.set_ylim(bottom=0.0, top=max(1.05, p_max * 1.05))
        ax.legend(fontsize="small", loc="upper right")
        self._canvas_p.draw_idle()

    def show_placeholder(self) -> None:
        """Clear all plots and display a waiting message."""
        self._draw_placeholder("Computing…")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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
        if message:
            self._ax_means.text(
                0.5,
                0.5,
                message,
                transform=self._ax_means.transAxes,
                ha="center",
                va="center",
                color="gray",
                fontsize=10,
            )
        self._canvas_means.draw_idle()
        self._canvas_t.draw_idle()
        self._canvas_p.draw_idle()

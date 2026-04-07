"""Middle panel: three tabbed matplotlib plots."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from gin_bids_py_analysis.processing.trial_slope_stats import TrialSlopeStatsProcessingResult
from gin_bids_py_analysis.processing.trial_stats import TrialStatsProcessingResult


class PlotPanel(QWidget):
    """Middle panel showing three tabbed plots for trial stats results.

    Tabs
    ----
    Activity      — Condition means with ±1 SEM shading; significance shown as a
                    thin bar at the bottom of the axes.
    T-values      — T-values over time with significance shading.
    P-values      — P-values with significance threshold and significant-interval shading.
    Trial Matrix  — Heatmap of individual trial epochs: condition A on top,
                    condition B below, separated by a white divider line.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(4, 4, 4, 4)

        self._tabs = QTabWidget()

        activity_w = QWidget()
        al = QVBoxLayout(activity_w)
        al.setContentsMargins(0, 0, 0, 0)
        self._fig_means = Figure(tight_layout=True)
        self._ax_means = self._fig_means.add_subplot(111)
        self._canvas_means = FigureCanvasQTAgg(self._fig_means)
        al.addWidget(self._canvas_means)
        self._tabs.addTab(activity_w, "Activity")

        t_w = QWidget()
        tl = QVBoxLayout(t_w)
        tl.setContentsMargins(0, 0, 0, 0)
        self._fig_t = Figure(tight_layout=True)
        self._ax_t = self._fig_t.add_subplot(111)
        self._canvas_t = FigureCanvasQTAgg(self._fig_t)
        tl.addWidget(self._canvas_t)
        self._tabs.addTab(t_w, "T-values")

        p_w = QWidget()
        pl = QVBoxLayout(p_w)
        pl.setContentsMargins(0, 0, 0, 0)
        self._fig_p = Figure(tight_layout=True)
        self._ax_p = self._fig_p.add_subplot(111)
        self._canvas_p = FigureCanvasQTAgg(self._fig_p)
        pl.addWidget(self._canvas_p)
        self._tabs.addTab(p_w, "P-values")

        matrix_w = QWidget()
        ml = QVBoxLayout(matrix_w)
        ml.setContentsMargins(0, 0, 0, 0)
        self._fig_matrix = Figure(tight_layout=True)
        self._ax_matrix = self._fig_matrix.add_subplot(111)
        self._canvas_matrix = FigureCanvasQTAgg(self._fig_matrix)
        ml.addWidget(self._canvas_matrix)
        self._tabs.addTab(matrix_w, "Trial Matrix")

        layout.addWidget(self._tabs)

        self._draw_placeholder("Select a subject and click Compute")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_plots(
        self,
        result: TrialStatsProcessingResult | TrialSlopeStatsProcessingResult,
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

        if _is_slope_result(result):
            self._tabs.setTabText(1, "Slopes")
            self._update_slope_plots(result, channel_idx)
        else:
            self._tabs.setTabText(1, "T-values")
            self._update_ttest_plots(result, channel_idx)

    def _update_ttest_plots(
        self,
        result: TrialStatsProcessingResult,
        channel_idx: int,
    ) -> None:
        ch = channel_idx
        t = result.time_axis_s
        ch_label = result.channel_names[ch]
        alpha = result.significance_alpha
        sig = result.significant_mask[ch].astype(bool)

        self._draw_activity_plot(
            t=t,
            ch_label=ch_label,
            condition_a=result.condition_a,
            condition_b=result.condition_b,
            condition_a_count=result.condition_a_trial_count,
            condition_b_count=result.condition_b_trial_count,
            mean_a=result.condition_a_mean[ch],
            mean_b=result.condition_b_mean[ch],
            sem_a=result.condition_a_sem[ch],
            sem_b=result.condition_b_sem[ch],
            sig_mask=sig,
        )

        ax = self._ax_t
        ax.clear()
        tv = result.t_values[ch]
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

        ax = self._ax_p
        ax.clear()
        p = result.p_values[ch]
        method = result.p_value_correction_method
        has_correction = bool(method) and method.lower() not in ("none", "")
        p_unc = (
            result.p_values_uncorrected[ch]
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
        ax.plot(t, p, color="purple", label=corr_label)
        ax.axhline(result.significance_alpha, color="red", linewidth=0.8, linestyle="--", label=f"α = {result.significance_alpha}")
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
        all_p = [p] if len(p) > 0 else []
        if p_unc is not None:
            all_p.append(p_unc)
        p_max = float(np.nanmax(np.concatenate(all_p))) if all_p else 1.0
        ax.set_ylim(bottom=0.0, top=max(1.05, p_max * 1.05))
        ax.legend(fontsize="small", loc="upper right")
        self._canvas_p.draw_idle()

        self._draw_trial_matrix(result, ch, t, ch_label)

    def _update_slope_plots(
        self,
        result: TrialSlopeStatsProcessingResult,
        channel_idx: int,
    ) -> None:
        ch = channel_idx
        t = result.time_axis_s
        ch_label = result.channel_names[ch]
        alpha = result.significance_alpha
        sig_a = result.condition_a_significant_mask[ch].astype(bool)
        sig_b = result.condition_b_significant_mask[ch].astype(bool)
        sig_any = sig_a | sig_b

        self._draw_activity_plot(
            t=t,
            ch_label=ch_label,
            condition_a=result.condition_a,
            condition_b=result.condition_b,
            condition_a_count=result.condition_a_trial_count,
            condition_b_count=result.condition_b_trial_count,
            mean_a=result.condition_a_mean[ch],
            mean_b=result.condition_b_mean[ch],
            sem_a=result.condition_a_sem[ch],
            sem_b=result.condition_b_sem[ch],
            sig_mask=sig_any,
        )

        ax = self._ax_t
        ax.clear()
        slope_a = result.condition_a_slope[ch]
        slope_b = result.condition_b_slope[ch]
        ax.plot(t, slope_a, color="steelblue", label=f"slope {result.condition_a}")
        ax.plot(t, slope_b, color="tomato", label=f"slope {result.condition_b}")
        if np.nanmin(np.concatenate([slope_a, slope_b])) < 0 < np.nanmax(np.concatenate([slope_a, slope_b])):
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        if sig_a.any():
            ax.fill_between(
                t,
                0.0,
                0.05,
                where=sig_a,
                alpha=0.25,
                color="steelblue",
                transform=ax.get_xaxis_transform(),
            )
        if sig_b.any():
            ax.fill_between(
                t,
                0.05,
                0.1,
                where=sig_b,
                alpha=0.25,
                color="tomato",
                transform=ax.get_xaxis_transform(),
            )
        ax.set_ylabel("slope")
        ax.legend(fontsize="small", loc="upper right")
        self._canvas_t.draw_idle()

        ax = self._ax_p
        ax.clear()
        p_a = result.condition_a_p_value_corrected[ch]
        p_b = result.condition_b_p_value_corrected[ch]
        p_a_raw = result.condition_a_p_value[ch]
        p_b_raw = result.condition_b_p_value[ch]
        method = result.p_value_correction_method
        has_correction = bool(method) and method.lower() not in ("none", "")
        if has_correction:
            ax.plot(t, p_a_raw, color="steelblue", linewidth=0.9, linestyle="--", alpha=0.55, label=f"p raw {result.condition_a}")
            ax.plot(t, p_b_raw, color="tomato", linewidth=0.9, linestyle="--", alpha=0.55, label=f"p raw {result.condition_b}")
        ax.plot(t, p_a, color="steelblue", label=f"p {result.condition_a}")
        ax.plot(t, p_b, color="tomato", label=f"p {result.condition_b}")
        ax.axhline(alpha, color="red", linewidth=0.8, linestyle="--", label=f"α = {alpha}")
        if sig_a.any():
            ax.fill_between(
                t,
                0,
                1,
                where=sig_a,
                alpha=0.1,
                color="steelblue",
                transform=ax.get_xaxis_transform(),
            )
        if sig_b.any():
            ax.fill_between(
                t,
                0,
                1,
                where=sig_b,
                alpha=0.1,
                color="tomato",
                transform=ax.get_xaxis_transform(),
            )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("p-value")
        all_p = [p_a, p_b]
        if has_correction:
            all_p.extend([p_a_raw, p_b_raw])
        p_max = float(np.nanmax(np.concatenate(all_p))) if all_p else 1.0
        ax.set_ylim(bottom=0.0, top=max(1.05, p_max * 1.05))
        ax.legend(fontsize="x-small", loc="upper right")
        self._canvas_p.draw_idle()

        self._draw_trial_matrix(result, ch, t, ch_label)

    def _draw_activity_plot(
        self,
        *,
        t: np.ndarray,
        ch_label: str,
        condition_a: str,
        condition_b: str,
        condition_a_count: int,
        condition_b_count: int,
        mean_a: np.ndarray,
        mean_b: np.ndarray,
        sem_a: np.ndarray,
        sem_b: np.ndarray,
        sig_mask: np.ndarray,
    ) -> None:
        ax = self._ax_means
        ax.clear()
        ax.plot(t, mean_a, color="steelblue", label=condition_a)
        ax.fill_between(t, mean_a - sem_a, mean_a + sem_a, alpha=0.25, color="steelblue")
        ax.plot(t, mean_b, color="tomato", label=condition_b)
        ax.fill_between(t, mean_b - sem_b, mean_b + sem_b, alpha=0.25, color="tomato")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel("Amplitude")
        ax.set_title(
            f"{ch_label}  —  {condition_a_count}× {condition_a} / {condition_b_count}× {condition_b}",
            fontsize=9,
        )
        if sig_mask.any():
            ax.fill_between(
                t,
                0.005,
                0.025,
                where=sig_mask,
                alpha=0.75,
                color="red",
                transform=ax.get_xaxis_transform(),
                zorder=5,
            )
        ax.legend(fontsize="small", loc="upper right")
        self._canvas_means.draw_idle()

    def _draw_trial_matrix(
        self,
        result: TrialStatsProcessingResult | TrialSlopeStatsProcessingResult,
        channel_idx: int,
        t: np.ndarray,
        ch_label: str,
    ) -> None:
        self._fig_matrix.clear()
        self._ax_matrix = self._fig_matrix.add_subplot(111)
        ax = self._ax_matrix
        epochs_a = result.condition_a_epochs
        epochs_b = result.condition_b_epochs
        n_a = epochs_a.shape[0] if epochs_a.ndim == 3 else 0
        n_b = epochs_b.shape[0] if epochs_b.ndim == 3 else 0
        if n_a == 0 and n_b == 0:
            ax.text(0.5, 0.5, "No epoch data", transform=ax.transAxes, ha="center", va="center", color="gray", fontsize=10)
        else:
            rows_a = epochs_a[:, channel_idx, :] if n_a > 0 else np.empty((0, len(t)))
            rows_b = epochs_b[:, channel_idx, :] if n_b > 0 else np.empty((0, len(t)))
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
            if n_a > 0:
                ax.axhline(n_a - 0.5, color="white", linewidth=1.5, linestyle="-")
            ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
            y_ticks = []
            y_labels = []
            if n_a > 0:
                y_ticks.append(n_a / 2 - 0.5)
                y_labels.append(result.condition_a)
            if n_b > 0:
                y_ticks.append(n_a + n_b / 2 - 0.5)
                y_labels.append(result.condition_b)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{ch_label}  —  trials ({n_a} / {n_b})", fontsize=9)
        self._canvas_matrix.draw_idle()

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
                color="gray",
                fontsize=10,
            )
        self._canvas_means.draw_idle()
        self._canvas_t.draw_idle()
        self._canvas_p.draw_idle()
        self._canvas_matrix.draw_idle()


def _is_slope_result(result: object) -> bool:
    return hasattr(result, "analysis_type") and getattr(result, "analysis_type", "") == "slope_regression"

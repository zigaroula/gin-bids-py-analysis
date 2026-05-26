"""Middle panel: tabbed matplotlib plots for subject-level trial results."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget
from scipy.stats import linregress

from gin_bids_py_analysis.processing.trial_stats import RegressionProcessingResult
from gin_bids_py_analysis.processing.trial_stats import ConditionTestProcessingResult

_SCATTER_SUMMARY_TARGET_BINS = 5


class PlotPanel(QWidget):
    """Middle panel showing tabbed plots for trial stats results.

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

        scatter_w = QWidget()
        sl = QVBoxLayout(scatter_w)
        sl.setContentsMargins(0, 0, 0, 0)
        self._fig_scatter = Figure(tight_layout=True)
        self._ax_scatter = self._fig_scatter.add_subplot(111)
        self._canvas_scatter = FigureCanvasQTAgg(self._fig_scatter)
        sl.addWidget(self._canvas_scatter)
        self._scatter_tab_index = self._tabs.addTab(scatter_w, "Scatter")
        self._tabs.setTabEnabled(self._scatter_tab_index, False)

        layout.addWidget(self._tabs)

        self._draw_placeholder("Select a subject and click Compute")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update_plots(
        self,
        result: ConditionTestProcessingResult | RegressionProcessingResult,
        channel_idx: int,
    ) -> None:
        """Redraw all three plots for the given channel index.

        If ``result.stats_valid`` is False, a notice is shown instead.
        """
        self._set_scatter_enabled(_is_slope_result(result))

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
        result: ConditionTestProcessingResult,
        channel_idx: int,
    ) -> None:
        ch = channel_idx
        t = result.time_axis_s
        ch_label = result.channel_names[ch]
        alpha = result.significance_alpha
        sig = result.contrast.significant_mask[ch].astype(bool)

        self._draw_activity_plot(
            t=t,
            ch_label=ch_label,
            condition_a=result.condition_a,
            condition_b=result.condition_b,
            condition_a_count=result.condition_a_trial_count,
            condition_b_count=result.condition_b_trial_count,
            mean_a=result.signal_activity.condition_a.mean[ch],
            mean_b=result.signal_activity.condition_b.mean[ch],
            sem_a=result.signal_activity.condition_a.sem[ch],
            sem_b=result.signal_activity.condition_b.sem[ch],
            sig_mask=sig,
            activity_zscore=result.activity_zscore,
        )

        ax = self._ax_t
        ax.clear()
        tv = result.contrast.t_values[ch]
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
        ax.set_title(
            self._format_channel_title(
                ch_label=ch_label,
                condition_a=result.condition_a,
                condition_b=result.condition_b,
                condition_a_count=result.condition_a_trial_count,
                condition_b_count=result.condition_b_trial_count,
                plot_label="t-values",
            ),
            fontsize=9,
        )
        _set_symmetric_ylim(ax)
        self._canvas_t.draw_idle()

        ax = self._ax_p
        ax.clear()
        p = result.contrast.p_values[ch]
        method = result.p_value_correction_method
        has_correction = bool(method) and method.lower() not in ("none", "")
        p_unc = (
            result.contrast.p_values_uncorrected[ch]
            if result.contrast.p_values_uncorrected.size > 0 and has_correction
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
        ax.set_title(
            self._format_channel_title(
                ch_label=ch_label,
                condition_a=result.condition_a,
                condition_b=result.condition_b,
                condition_a_count=result.condition_a_trial_count,
                condition_b_count=result.condition_b_trial_count,
                plot_label="p-values",
            ),
            fontsize=9,
        )
        all_p = [p] if len(p) > 0 else []
        if p_unc is not None:
            all_p.append(p_unc)
        p_max = float(np.nanmax(np.concatenate(all_p))) if all_p else 1.0
        ax.set_ylim(bottom=0.0, top=max(1.05, p_max * 1.05))
        ax.legend(fontsize="small", loc="upper right")
        self._canvas_p.draw_idle()

        self._draw_trial_matrix(result, ch, t, ch_label)
        self._draw_scatter_placeholder("Scatter available only in slope mode")

    def _update_slope_plots(
        self,
        result: RegressionProcessingResult,
        channel_idx: int,
    ) -> None:
        ch = channel_idx
        t = result.time_axis_s
        ch_label = result.channel_names[ch]
        alpha = result.significance_alpha
        sig_a = result.regression.condition_a.significant_mask[ch].astype(bool)
        sig_b = result.regression.condition_b.significant_mask[ch].astype(bool)
        sig_any = sig_a | sig_b

        self._draw_activity_plot(
            t=t,
            ch_label=ch_label,
            condition_a=result.condition_a,
            condition_b=result.condition_b,
            condition_a_count=result.condition_a_trial_count,
            condition_b_count=result.condition_b_trial_count,
            mean_a=result.signal_activity.condition_a.mean[ch],
            mean_b=result.signal_activity.condition_b.mean[ch],
            sem_a=result.signal_activity.condition_a.sem[ch],
            sem_b=result.signal_activity.condition_b.sem[ch],
            sig_mask=sig_any,
            activity_zscore=result.activity_zscore,
        )

        ax = self._ax_t
        ax.clear()
        slope_a = result.regression.condition_a.slope[ch]
        slope_b = result.regression.condition_b.slope[ch]
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
        ax.set_title(
            self._format_channel_title(
                ch_label=ch_label,
                condition_a=result.condition_a,
                condition_b=result.condition_b,
                condition_a_count=result.condition_a_trial_count,
                condition_b_count=result.condition_b_trial_count,
                plot_label="slopes",
            ),
            fontsize=9,
        )
        ax.legend(fontsize="small", loc="upper right")
        _set_symmetric_ylim(ax)
        self._canvas_t.draw_idle()

        ax = self._ax_p
        ax.clear()
        p_a = result.regression.condition_a.p_value_corrected[ch]
        p_b = result.regression.condition_b.p_value_corrected[ch]
        p_a_raw = result.regression.condition_a.p_value[ch]
        p_b_raw = result.regression.condition_b.p_value[ch]
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
        ax.set_title(
            self._format_channel_title(
                ch_label=ch_label,
                condition_a=result.condition_a,
                condition_b=result.condition_b,
                condition_a_count=result.condition_a_trial_count,
                condition_b_count=result.condition_b_trial_count,
                plot_label="slope p-values",
            ),
            fontsize=9,
        )
        all_p = [p_a, p_b]
        if has_correction:
            all_p.extend([p_a_raw, p_b_raw])
        p_max = float(np.nanmax(np.concatenate(all_p))) if all_p else 1.0
        ax.set_ylim(bottom=0.0, top=max(1.05, p_max * 1.05))
        ax.legend(fontsize="x-small", loc="upper right")
        self._canvas_p.draw_idle()

        self._draw_trial_matrix(result, ch, t, ch_label)
        self._draw_scatter_plot(result, ch, ch_label)

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
        activity_zscore: str,
    ) -> None:
        ax = self._ax_means
        ax.clear()
        ax.plot(t, mean_a, color="steelblue", label=condition_a)
        ax.fill_between(t, mean_a - sem_a, mean_a + sem_a, alpha=0.25, color="steelblue")
        ax.plot(t, mean_b, color="tomato", label=condition_b)
        ax.fill_between(t, mean_b - sem_b, mean_b + sem_b, alpha=0.25, color="tomato")
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_ylabel(_activity_axis_label(activity_zscore))
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
        _set_symmetric_ylim(ax)
        self._canvas_means.draw_idle()

    def _draw_trial_matrix(
        self,
        result: ConditionTestProcessingResult | RegressionProcessingResult,
        channel_idx: int,
        t: np.ndarray,
        ch_label: str,
    ) -> None:
        self._fig_matrix.clear()
        self._ax_matrix = self._fig_matrix.add_subplot(111)
        ax = self._ax_matrix
        epochs_a = result.epochs.condition_a
        epochs_b = result.epochs.condition_b
        n_a = epochs_a.shape[0] if epochs_a.ndim == 3 else 0
        n_b = epochs_b.shape[0] if epochs_b.ndim == 3 else 0
        shown_a = 0
        shown_b = 0
        masked_a = 0
        masked_b = 0
        if n_a == 0 and n_b == 0:
            ax.text(0.5, 0.5, "No epoch data", transform=ax.transAxes, ha="center", va="center", color="gray", fontsize=10)
        else:
            rows_a = epochs_a[:, channel_idx, :] if n_a > 0 else np.empty((0, len(t)))
            rows_b = epochs_b[:, channel_idx, :] if n_b > 0 else np.empty((0, len(t)))
            rows_a, _ = _filter_matrix_rows(rows_a)
            rows_b, _ = _filter_matrix_rows(rows_b)
            shown_a = int(rows_a.shape[0])
            shown_b = int(rows_b.shape[0])
            masked_a = int(n_a - shown_a)
            masked_b = int(n_b - shown_b)
            if shown_a == 0 and shown_b == 0:
                ax.text(
                    0.5,
                    0.5,
                    "No finite trial data for selected feature",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    color="gray",
                    fontsize=10,
                )
            else:
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
                    extent=[t[0], t[-1], shown_a + shown_b - 0.5, -0.5],
                    interpolation="nearest",
                )
                self._fig_matrix.colorbar(im, ax=ax, location="right", shrink=0.8)
                if shown_a > 0 and shown_b > 0:
                    ax.axhline(shown_a - 0.5, color="white", linewidth=1.5, linestyle="-")
            ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
            y_ticks = []
            y_labels = []
            if shown_a > 0:
                y_ticks.append(shown_a / 2 - 0.5)
                y_labels.append(result.condition_a)
            if shown_b > 0:
                y_ticks.append(shown_a + shown_b / 2 - 0.5)
                y_labels.append(result.condition_b)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_xlabel("Time (s)")
        ax.set_title(f"{ch_label}  —  trials ({n_a} / {n_b})", fontsize=9)
        ax.set_title(
            _trial_matrix_title(
                ch_label=ch_label,
                shown_a=shown_a,
                shown_b=shown_b,
                masked_a=masked_a,
                masked_b=masked_b,
            ),
            fontsize=9,
        )
        self._canvas_matrix.draw_idle()

    def _draw_scatter_plot(
        self,
        result: RegressionProcessingResult,
        channel_idx: int,
        ch_label: str,
    ) -> None:
        self._fig_scatter.clear()
        self._ax_scatter = self._fig_scatter.add_subplot(111)
        ax = self._ax_scatter

        pred_a, act_a, invalid_a = self._extract_scatter_series(
            predictor_values=result.predictor_values.condition_a.values,
            summary_values=result.trial_activity_summary_values.condition_a,
            channel_idx=channel_idx,
        )
        pred_b, act_b, invalid_b = self._extract_scatter_series(
            predictor_values=result.predictor_values.condition_b.values,
            summary_values=result.trial_activity_summary_values.condition_b,
            channel_idx=channel_idx,
        )

        if invalid_a or invalid_b:
            self._draw_scatter_placeholder("No scatter data available")
            return

        plotted = False
        plotted |= self._plot_scatter_condition(
            ax=ax,
            predictor_values=pred_a,
            activity_values=act_a,
            color="steelblue",
            label=result.condition_a,
        )
        plotted |= self._plot_scatter_condition(
            ax=ax,
            predictor_values=pred_b,
            activity_values=act_b,
            color="tomato",
            label=result.condition_b,
        )
        if not plotted:
            self._draw_scatter_placeholder("No scatter data available")
            return

        ax.set_xlabel(_predictor_axis_label(result.predictor, result.predictor_zscore))
        ax.set_ylabel(
            _scatter_activity_axis_label(
                result.trial_activity_summary_label,
                result.activity_zscore,
            )
        )
        ax.set_title(
            self._format_channel_title(
                ch_label=ch_label,
                condition_a=result.condition_a,
                condition_b=result.condition_b,
                condition_a_count=result.condition_a_trial_count,
                condition_b_count=result.condition_b_trial_count,
                plot_label="predictor vs activity",
            ),
            fontsize=9,
        )
        _safe_legend(ax)
        self._canvas_scatter.draw_idle()

    def _draw_scatter_placeholder(self, message: str) -> None:
        self._fig_scatter.clear()
        self._ax_scatter = self._fig_scatter.add_subplot(111)
        self._ax_scatter.set_facecolor("#f4f4f4")
        self._ax_scatter.set_xticks([])
        self._ax_scatter.set_yticks([])
        self._ax_scatter.text(
            0.5,
            0.5,
            message,
            transform=self._ax_scatter.transAxes,
            ha="center",
            va="center",
            color="gray",
            fontsize=10,
        )
        self._canvas_scatter.draw_idle()

    def _extract_scatter_series(
        self,
        *,
        predictor_values: np.ndarray,
        summary_values: np.ndarray,
        channel_idx: int,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        predictor = np.asarray(predictor_values, dtype=np.float64).ravel()
        activity = np.asarray(summary_values, dtype=np.float64)

        if predictor.size == 0 and activity.size == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), False
        if predictor.size == 0 or activity.size == 0:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), True
        if activity.ndim != 2 or channel_idx < 0 or channel_idx >= activity.shape[0]:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), True

        activity_row = np.asarray(activity[channel_idx], dtype=np.float64).ravel()
        if activity_row.size != predictor.size:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64), True

        return predictor, activity_row, False

    def _plot_scatter_condition(
        self,
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
        self._fig_scatter.clear()
        self._ax_scatter = self._fig_scatter.add_subplot(111)
        self._ax_scatter.set_facecolor("#f4f4f4")
        self._ax_scatter.set_xticks([])
        self._ax_scatter.set_yticks([])
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
        self._canvas_scatter.draw_idle()

    def _set_scatter_enabled(self, enabled: bool) -> None:
        self._tabs.setTabEnabled(self._scatter_tab_index, enabled)
        if not enabled and self._tabs.currentIndex() == self._scatter_tab_index:
            self._tabs.setCurrentIndex(0)

    def _format_channel_title(
        self,
        *,
        ch_label: str,
        condition_a: str,
        condition_b: str,
        condition_a_count: int,
        condition_b_count: int,
        plot_label: str,
    ) -> str:
        return (
            f"{ch_label} - {plot_label} - "
            f"{condition_a_count} x {condition_a} / {condition_b_count} x {condition_b}"
        )


def _is_slope_result(result: object) -> bool:
    return hasattr(result, "analysis_type") and getattr(result, "analysis_type", "") == "slope_regression"


def _set_symmetric_ylim(ax) -> None:
    """Make the y-axis limits symmetric around zero: [-max_abs, +max_abs]."""
    ymin, ymax = ax.get_ylim()
    bound = max(abs(ymin), abs(ymax))
    if bound > 0:
        ax.set_ylim(-bound, bound)


def _safe_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend(fontsize="small", loc="upper right")


def _compute_scatter_summary_points(
    predictor_values: np.ndarray,
    activity_values: np.ndarray,
    *,
    target_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    predictor = np.asarray(predictor_values, dtype=np.float64).ravel()
    activity = np.asarray(activity_values, dtype=np.float64).ravel()
    valid = np.isfinite(predictor) & np.isfinite(activity)
    if valid.sum() < 4:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty

    predictor = predictor[valid]
    activity = activity[valid]

    n_bins = min(int(target_bins), int(predictor.size))
    if n_bins < 2:
        empty = np.empty(0, dtype=np.float64)
        return empty, empty, empty, empty

    mean_x: list[float] = []
    mean_y: list[float] = []
    sem_x: list[float] = []
    sem_y: list[float] = []
    edges = np.quantile(predictor, np.linspace(0.0, 1.0, n_bins + 1))

    for lower, upper in zip(edges[:-1], edges[1:]):
        in_bin = (predictor >= lower) & (predictor <= upper)
        predictor_bin = predictor[in_bin]
        activity_bin = activity[in_bin]
        if predictor_bin.size == 0:
            continue
        mean_x.append(float(np.nanmean(predictor_bin)))
        mean_y.append(float(np.nanmean(activity_bin)))
        sem_x.append(_sem_or_zero(predictor_bin))
        sem_y.append(_sem_or_zero(activity_bin))

    return (
        np.asarray(mean_x, dtype=np.float64),
        np.asarray(mean_y, dtype=np.float64),
        np.asarray(sem_x, dtype=np.float64),
        np.asarray(sem_y, dtype=np.float64),
    )


def _sem_or_zero(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64).ravel()
    finite = arr[np.isfinite(arr)]
    if finite.size < 2:
        return 0.0
    sem = float(np.nanstd(finite, ddof=1) / np.sqrt(finite.size))
    return sem if np.isfinite(sem) else 0.0


def _activity_axis_label(activity_zscore: str) -> str:
    if _is_activity_zscore_enabled(activity_zscore):
        return "Activity (z)"
    return "Amplitude"


def _scatter_activity_axis_label(summary_label: str, activity_zscore: str) -> str:
    label = str(summary_label).strip() or "Epoch mean activity"
    if _is_activity_zscore_enabled(activity_zscore):
        return f"{label} (z)"
    return label


def _predictor_axis_label(predictor: str, predictor_zscore: str) -> str:
    base = predictor.strip() if predictor else "Predictor value"
    if str(predictor_zscore).strip().lower() in {"condition", "global"}:
        return f"{base} (z)"
    return base


def _is_activity_zscore_enabled(activity_zscore: str) -> bool:
    return str(activity_zscore).strip().lower() in {"baseline", "across_trials"}


def _filter_matrix_rows(rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("rows must be 2-D (n_rows, n_times).")
    if arr.shape[0] == 0:
        return arr, np.zeros((0,), dtype=bool)
    keep = np.any(np.isfinite(arr), axis=1)
    return arr[keep], keep


def _trial_matrix_title(
    *,
    ch_label: str,
    shown_a: int,
    shown_b: int,
    masked_a: int,
    masked_b: int,
) -> str:
    if masked_a > 0 or masked_b > 0:
        return (
            f"{ch_label} - trials shown ({shown_a} / {shown_b}), "
            f"masked ({masked_a} / {masked_b})"
        )
    return f"{ch_label} - trials ({shown_a} / {shown_b})"


class _ScatterRegressionResult:
    def __init__(self, *, slope: float, intercept: float, p_value: float) -> None:
        self.slope = float(slope)
        self.intercept = float(intercept)
        self.p_value = float(p_value)

    @property
    def is_significant(self) -> bool:
        return np.isfinite(self.p_value) and self.p_value < 0.05


def _fit_scatter_regression(
    predictor_values: np.ndarray,
    activity_values: np.ndarray,
) -> _ScatterRegressionResult | None:
    predictor = np.asarray(predictor_values, dtype=np.float64).ravel()
    activity = np.asarray(activity_values, dtype=np.float64).ravel()
    valid = np.isfinite(predictor) & np.isfinite(activity)
    predictor = predictor[valid]
    activity = activity[valid]
    if predictor.size < 2:
        return None
    if np.nanmin(predictor) == np.nanmax(predictor):
        return None
    try:
        fit = linregress(predictor, activity)
    except ValueError:
        return None
    if not np.isfinite(fit.slope) or not np.isfinite(fit.intercept):
        return None
    return _ScatterRegressionResult(
        slope=float(fit.slope),
        intercept=float(fit.intercept),
        p_value=float(fit.pvalue),
    )



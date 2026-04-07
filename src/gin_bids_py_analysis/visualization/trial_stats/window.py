"""Main application window for trial statistics visualization."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QTabWidget,
    QWidget,
)

from gin_bids_py_analysis.bids import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_slope_stats import (
    TrialSlopeStatsParams,
    TrialSlopeStatsProcessing,
    TrialSlopeStatsProcessingResult,
    TrialSlopeStatsProcessingWriter,
    TrialSlopeStatsWriterParams,
)
from gin_bids_py_analysis.processing.trial_slope_stats_group import (
    TrialSlopeStatsGroupParams,
    TrialSlopeStatsGroupProcessingResult,
    TrialSlopeStatsGroupProcessingWriter,
    TrialSlopeStatsGroupWriterParams,
)
from gin_bids_py_analysis.processing.trial_stats import (
    TrialResolver,
    TrialStatsParams,
    TrialStatsProcessing,
    TrialStatsProcessingResult,
    TrialStatsProcessingWriter,
)

from .panels.group_params_panel import GroupParamsPanel
from .panels.group_plot_panel import GroupPlotPanel
from .panels.params_panel import ParamsPanel
from .panels.plot_panel import PlotPanel
from .panels.save_dialog import SaveTrialStatsDialog
from .panels.save_group_dialog import SaveTrialStatsGroupDialog
from .panels.subject_panel import SubjectChannelPanel
from .worker import ComputeAllWorker, GroupComputeWorker, PreloadWorker, WriteAllWorker, WriteGroupWorker

if TYPE_CHECKING:
    from gin_bids_py_analysis.processing.trial_stats_group.params import (
        TrialStatsGroupParams,
    )
    from gin_bids_py_analysis.processing.trial_stats_group.result import (
        TrialStatsGroupProcessingResult,
    )


class TrialStatsWindow(QMainWindow):
    """Trial statistics visualization window with Subject and Group tabs.

    Subject tab (left → middle → right)
    ------------------------------------
    ``SubjectChannelPanel`` | ``PlotPanel`` | ``ParamsPanel``

    Group tab (left/middle → right)
    --------------------------------
    ``GroupPlotPanel`` (with built-in ROI list) | ``GroupParamsPanel``

    Pressing **Compute** runs all subjects at once.  The Group tab is
    enabled only after every subject has a result and *group_params* is
    provided.

    Parameters
    ----------
    subject_groups:
        Mapping of subject id to ``BIDSFileGroup`` for that subject.
    default_params:
        Initial ``TrialStatsParams`` to pre-populate the parameters panel.
    resolver:
        The ``TrialResolver`` configured for this dataset.
    group_params:
        Optional ``TrialStatsGroupParams``.  When ``None`` the Group tab is
        visible but permanently disabled.
    bids_root:
        Optional path to the BIDS dataset root.  When provided the *Save results*
        buttons in both panels become functional after a successful compute.
    """

    def __init__(
        self,
        subject_groups: dict[str, BIDSFileGroup],
        default_params: TrialStatsParams,
        resolver: TrialResolver,
        group_params: "TrialStatsGroupParams | None" = None,
        slope_group_params: TrialSlopeStatsGroupParams | None = None,
        bids_root: Path | None = None,
        default_slope_params: TrialSlopeStatsParams | None = None,
        default_mode: str = "ttest",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Trial Statistics Viewer")
        self.resize(1280, 760)

        self._subject_groups = subject_groups
        self._resolver = resolver
        self._bids_root = bids_root
        self._current_result: TrialStatsProcessingResult | TrialSlopeStatsProcessingResult | None = None
        self._current_worker: ComputeAllWorker | None = None
        # Guard variable: only accept results whose generation matches this value
        self._compute_generation: int = 0
        self._preload_worker: PreloadWorker | None = None
        self._group_params = group_params
        self._slope_group_params = slope_group_params
        self._group_result: TrialStatsGroupProcessingResult | TrialSlopeStatsGroupProcessingResult | None = None
        self._group_worker: GroupComputeWorker | None = None
        self._write_worker: WriteAllWorker | None = None
        self._write_group_worker: WriteGroupWorker | None = None
        # Accumulated per-subject results; unlocks Group tab when all are done
        self._all_results: dict[str, TrialStatsProcessingResult | TrialSlopeStatsProcessingResult] = {}
        self._analysis_mode = default_mode if default_mode in {"ttest", "slope"} else "ttest"

        subject_ids = sorted(subject_groups.keys())

        # ------------------------------------------------------------------
        # Subject tab panels
        # ------------------------------------------------------------------
        self._subject_panel = SubjectChannelPanel(subject_ids)
        self._plot_panel = PlotPanel()
        self._params_panel = ParamsPanel(
            default_params,
            slope_params=default_slope_params,
            default_mode=self._analysis_mode,
        )

        subject_splitter = QSplitter(Qt.Orientation.Horizontal)
        subject_splitter.addWidget(self._subject_panel)
        subject_splitter.addWidget(self._plot_panel)
        subject_splitter.addWidget(self._params_panel)
        subject_splitter.setSizes([200, 740, 340])
        subject_splitter.setStretchFactor(0, 0)
        subject_splitter.setStretchFactor(1, 1)
        subject_splitter.setStretchFactor(2, 0)

        subject_tab = QWidget()
        subject_layout = QHBoxLayout(subject_tab)
        subject_layout.setContentsMargins(0, 0, 0, 0)
        subject_layout.addWidget(subject_splitter)

        # ------------------------------------------------------------------
        # Group tab panels
        # ------------------------------------------------------------------
        self._group_plot_panel = GroupPlotPanel()
        if self._analysis_mode == "slope":
            active_group_params: TrialStatsGroupParams | TrialSlopeStatsGroupParams
            if self._slope_group_params is not None:
                active_group_params = self._slope_group_params
            elif self._group_params is not None:
                active_group_params = _coerce_slope_group_params_from_ttest(self._group_params)
            else:
                active_group_params = _make_placeholder_slope_group_params()
            self._group_params_panel = GroupParamsPanel(
                active_group_params,
                subject_ids=subject_ids,
                analysis_mode="slope",
            )
        else:
            if self._group_params is not None:
                active_group_params = self._group_params
            else:
                active_group_params = _make_placeholder_group_params()
            self._group_params_panel = GroupParamsPanel(
                active_group_params,
                subject_ids=subject_ids,
                analysis_mode="ttest",
            )

        group_splitter = QSplitter(Qt.Orientation.Horizontal)
        group_splitter.addWidget(self._group_plot_panel)
        group_splitter.addWidget(self._group_params_panel)
        group_splitter.setSizes([940, 340])
        group_splitter.setStretchFactor(0, 1)
        group_splitter.setStretchFactor(1, 0)

        group_tab = QWidget()
        group_layout = QHBoxLayout(group_tab)
        group_layout.setContentsMargins(0, 0, 0, 0)
        group_layout.addWidget(group_splitter)

        # ------------------------------------------------------------------
        # Tab widget
        # ------------------------------------------------------------------
        self._tabs = QTabWidget()
        self._tabs.addTab(subject_tab, "Subject")
        self._tabs.addTab(group_tab, "Group")
        self.setCentralWidget(self._tabs)

        # Group tab starts disabled until all subjects are computed
        self._tabs.setTabEnabled(1, False)

        # ------------------------------------------------------------------
        # Wire signals — Subject tab
        # ------------------------------------------------------------------
        self._subject_panel.subject_changed.connect(self._on_subject_changed)
        self._subject_panel.channel_changed.connect(self._on_channel_changed)
        self._params_panel.compute_requested.connect(self._on_compute_requested)

        # Wire signals — Group tab
        self._group_params_panel.compute_requested.connect(self._on_group_compute_requested)

        # Wire save signals
        self._params_panel.save_requested.connect(self._on_save_requested)
        self._group_params_panel.save_requested.connect(self._on_group_save_requested)

        # Pre-load all files in the background; auto-compute all subjects once done
        if subject_ids:
            QTimer.singleShot(0, self._start_preload)

    # ------------------------------------------------------------------
    # Subject tab slot handlers
    # ------------------------------------------------------------------

    def _trigger_initial_compute(self) -> None:
        self._start_compute_all()

    def _on_subject_changed(self, subject_id: str) -> None:
        # Show a cached result immediately if available; otherwise re-run all
        if subject_id in self._all_results:
            self._display_subject_result(subject_id, self._all_results[subject_id])
        else:
            try:
                mode, params = self._params_panel.get_mode_and_params()
            except ValueError:
                return
            self._all_results.clear()
            self._tabs.setTabEnabled(1, False)
            self._start_compute_all(params=params, mode=mode)

    def _on_channel_changed(self, channel_idx: int) -> None:
        if self._current_result is not None and channel_idx >= 0:
            self._plot_panel.update_plots(self._current_result, channel_idx)

    def _on_compute_requested(self) -> None:
        try:
            mode, params = self._params_panel.get_mode_and_params()
        except ValueError:
            return
        self._all_results.clear()
        self._params_panel.set_save_enabled(False)
        self._tabs.setTabEnabled(1, False)
        self._start_compute_all(params=params, mode=mode)

    # ------------------------------------------------------------------
    # Pre-loading lifecycle
    # ------------------------------------------------------------------

    def _start_preload(self) -> None:
        total_files = sum(
            len(group.all_files) for group in self._subject_groups.values()
        )
        self._subject_panel.set_interactive(False)
        self._params_panel.set_computing(True)
        self._params_panel.set_status(
            f"Pre-loading {total_files} file(s) for {len(self._subject_groups)} subject(s)…"
        )

        worker = PreloadWorker(self._subject_groups, parent=self)
        self._preload_worker = worker
        worker.progress.connect(self._params_panel.set_status)
        worker.finished.connect(self._on_preload_finished)
        worker.error.connect(self._on_preload_error)
        worker.start()

    def _on_preload_finished(self) -> None:
        self._subject_panel.set_interactive(True)
        self._params_panel.set_computing(False)
        self._params_panel.set_status("Files loaded — ready")
        self._trigger_initial_compute()

    def _on_preload_error(self, message: str) -> None:
        self._subject_panel.set_interactive(True)
        self._params_panel.set_computing(False)
        self._params_panel.set_status(f"Pre-load error: {message}")
        # Fall back to on-demand loading so the user can still compute
        self._trigger_initial_compute()

    # ------------------------------------------------------------------
    # "Compute all subjects" lifecycle
    # ------------------------------------------------------------------

    def _start_compute_all(
        self,
        params: TrialStatsParams | TrialSlopeStatsParams | None = None,
        mode: str | None = None,
    ) -> None:
        if params is None:
            try:
                mode, params = self._params_panel.get_mode_and_params()
            except ValueError:
                return
        if mode is None:
            mode = self._params_panel.analysis_mode
        self._analysis_mode = mode if mode in {"ttest", "slope"} else "ttest"
        self._group_params_panel.set_analysis_mode(
            "slope" if self._analysis_mode == "slope" else "ttest"
        )
        if self._analysis_mode == "slope":
            if self._slope_group_params is not None:
                self._group_params_panel.set_params(self._slope_group_params)
            elif self._group_params is not None:
                self._group_params_panel.set_params(
                    _coerce_slope_group_params_from_ttest(self._group_params)
                )
        elif self._group_params is not None:
            self._group_params_panel.set_params(self._group_params)

        # Bump generation so any in-flight result from a previous run is discarded
        self._compute_generation += 1
        current_gen = self._compute_generation

        # Disconnect previous worker
        if self._current_worker is not None:
            try:
                self._current_worker.subject_done.disconnect()
                self._current_worker.all_done.disconnect()
                self._current_worker.progress.disconnect()
                self._current_worker.error.disconnect()
            except RuntimeError:
                pass
            self._current_worker = None

        self._plot_panel.show_placeholder()
        self._params_panel.set_save_enabled(False)
        self._params_panel.set_computing(True)
        self._params_panel.set_status(
            f"Computing {len(self._subject_groups)} subject(s)…"
        )

        resolver = self._resolver
        if self._analysis_mode == "slope":
            processor_factory = (
                lambda p: TrialSlopeStatsProcessing(
                    p,  # type: ignore[arg-type]
                    resolver=resolver,
                )
            )
        else:
            processor_factory = (
                lambda p: TrialStatsProcessing(
                    p,  # type: ignore[arg-type]
                    resolver=resolver,
                )
            )
        worker = ComputeAllWorker(
            processor_factory=processor_factory,
            subject_groups=self._subject_groups,
            params=params,
            parent=self,
        )
        self._current_worker = worker

        worker.subject_done.connect(
            lambda sid, res, gen=current_gen: self._on_subject_done(sid, res, gen)
        )
        worker.all_done.connect(
            lambda results, gen=current_gen: self._on_all_done(results, gen)
        )
        worker.progress.connect(self._params_panel.set_status)
        worker.error.connect(
            lambda msg, gen=current_gen: self._on_compute_error(msg, gen)
        )
        worker.start()

    def _on_subject_done(
        self,
        subject_id: str,
        result: TrialStatsProcessingResult | TrialSlopeStatsProcessingResult,
        generation: int,
    ) -> None:
        if generation != self._compute_generation:
            return  # stale

        self._all_results[subject_id] = result

        # If this is the currently shown subject, update the Subject tab immediately
        if subject_id == self._subject_panel.current_subject:
            self._display_subject_result(subject_id, result)

    def _on_all_done(
        self,
        results: dict[str, TrialStatsProcessingResult | TrialSlopeStatsProcessingResult],
        generation: int,
    ) -> None:
        if generation != self._compute_generation:
            return  # stale

        self._params_panel.set_computing(False)
        n = len(results)
        self._params_panel.set_status(f"Done — {n} subject(s) computed")

        # Enable Save button when there are results and a bids_root is set
        if n > 0 and self._bids_root is not None:
            self._params_panel.set_save_enabled(True)

        # Ensure the current subject's result is displayed
        current = self._subject_panel.current_subject
        if current in results and self._current_result is not results.get(current):
            self._display_subject_result(current, results[current])

        if self._analysis_mode == "slope":
            if self._slope_group_params is not None or self._group_params is not None:
                self._tabs.setTabEnabled(1, True)
                self._group_params_panel.set_status(
                    f"{n} subject(s) ready — click 'Compute group stats'"
                )
            else:
                self._tabs.setTabEnabled(1, False)
                self._group_params_panel.set_status(
                    "Set slope-group parameters to enable Group compute."
                )
        elif self._group_params is not None and n > 0:
            self._tabs.setTabEnabled(1, True)
            self._group_params_panel.set_status(
                f"{n} subject(s) ready — click 'Compute group stats'"
            )

    def _on_compute_error(self, message: str, generation: int) -> None:
        if generation != self._compute_generation:
            return  # stale
        self._params_panel.set_computing(False)
        self._params_panel.set_status(f"Error: {message}")

    def _display_subject_result(
        self,
        subject_id: str,
        result: TrialStatsProcessingResult | TrialSlopeStatsProcessingResult,
    ) -> None:
        """Update Subject tab plots for *result* (already in memory)."""
        self._current_result = result
        n_ok = len(self._all_results)
        n_total = len(self._subject_groups)
        self._params_panel.set_status(
            f"[{n_ok}/{n_total}] {subject_id} — "
            f"{result.condition_a_trial_count}× {result.condition_a} / "
            f"{result.condition_b_trial_count}× {result.condition_b}"
        )
        previous_channel = self._subject_panel.current_channel_name
        self._subject_panel.set_channels(
            result.channel_names,
            restore_name=previous_channel,
            channel_significant_mask=getattr(result, "channel_significant_mask", None),
        )
        self._plot_panel.update_plots(result, self._subject_panel.current_channel_index)

    # ------------------------------------------------------------------
    # Group tab lifecycle
    # ------------------------------------------------------------------

    def _on_group_compute_requested(self) -> None:
        if not self._all_results:
            self._group_params_panel.set_status("No subject results available yet.")
            return
        try:
            params = self._group_params_panel.get_params()
        except ValueError:
            return
        if self._analysis_mode == "slope":
            if isinstance(params, TrialSlopeStatsGroupParams):
                self._slope_group_params = params
            else:
                self._slope_group_params = _coerce_slope_group_params_from_ttest(params)
                params = self._slope_group_params
        else:
            assert isinstance(params, TrialStatsGroupParams)
            self._group_params = params
        self._start_group_compute(params)

    def _start_group_compute(
        self,
        params: TrialStatsGroupParams | TrialSlopeStatsGroupParams,
    ) -> None:
        # Disconnect previous group worker
        if self._group_worker is not None:
            try:
                self._group_worker.result_ready.disconnect()
                self._group_worker.error.disconnect()
            except RuntimeError:
                pass
            self._group_worker = None

        self._group_plot_panel.show_placeholder()
        self._group_params_panel.set_save_enabled(False)
        self._group_params_panel.set_computing(True)
        self._group_params_panel.set_status(
            f"Computing group stats for {len(self._all_results)} subject(s)…"
        )

        worker = GroupComputeWorker(self._all_results, params, parent=self)
        self._group_worker = worker
        worker.result_ready.connect(self._on_group_result_ready)
        worker.error.connect(self._on_group_error)
        worker.start()

    def _on_group_result_ready(
        self,
        result: TrialStatsGroupProcessingResult | TrialSlopeStatsGroupProcessingResult,
    ) -> None:
        self._group_result = result
        self._group_params_panel.set_computing(False)
        n_rois = len(result.region_names)
        excluded = len(result.excluded_rois)
        status = f"Done — {n_rois} ROI(s)"
        if excluded:
            status += f" ({excluded} excluded)"
        self._group_params_panel.set_status(status)
        if self._bids_root is not None:
            self._group_params_panel.set_save_enabled(True)
        self._group_plot_panel.update_plots(result, 0)

    def _on_group_error(self, message: str) -> None:
        self._group_params_panel.set_computing(False)
        self._group_params_panel.set_status(f"Error: {message}")

    # ------------------------------------------------------------------
    # Save lifecycle
    # ------------------------------------------------------------------

    def _on_save_requested(self) -> None:
        if not self._all_results:
            QMessageBox.information(
                self, "Nothing to save", "Run Compute first before saving."
            )
            return
        if self._bids_root is None:
            QMessageBox.warning(
                self,
                "No BIDS root",
                "No BIDS root was provided when the viewer was launched. "
                "Cannot determine where to write output files.",
            )
            return

        default_pipeline_label = (
            "trial_slope_stats" if self._analysis_mode == "slope" else "trial_stats"
        )
        default_output_description = (
            "trialslopestats" if self._analysis_mode == "slope" else "trialstats"
        )
        dlg = SaveTrialStatsDialog(
            self._bids_root,
            parent=self,
            default_pipeline_label=default_pipeline_label,
            default_output_description=default_output_description,
        )
        if dlg.exec() != SaveTrialStatsDialog.DialogCode.Accepted:
            return

        if self._analysis_mode == "slope":
            writer_params = TrialSlopeStatsWriterParams(**dlg.get_common_writer_kwargs())
            writer = TrialSlopeStatsProcessingWriter(writer_params)
        else:
            writer_params = dlg.get_writer_params()
            writer = TrialStatsProcessingWriter(writer_params)

        n = len(self._all_results)
        self._params_panel.set_save_enabled(False)
        self._params_panel.set_status(f"Saving {n} subject(s)…")

        if self._write_worker is not None:
            try:
                self._write_worker.finished.disconnect()
                self._write_worker.error.disconnect()
                self._write_worker.progress.disconnect()
            except RuntimeError:
                pass

        worker = WriteAllWorker(dict(self._all_results), writer, parent=self)
        self._write_worker = worker
        worker.progress.connect(self._params_panel.set_status)
        worker.finished.connect(self._on_write_finished)
        worker.error.connect(self._on_write_error)
        worker.start()

    def _on_write_finished(self, n: int) -> None:
        subject_word = "subject" if n == 1 else "subjects"
        self._params_panel.set_status(f"Saved {n} {subject_word}")
        self._params_panel.set_save_enabled(True)

    def _on_write_error(self, message: str) -> None:
        self._params_panel.set_save_enabled(True)
        self._params_panel.set_status(f"Save error: {message}")
        QMessageBox.critical(self, "Save error", message)

    def _on_group_save_requested(self) -> None:
        if self._group_result is None:
            QMessageBox.information(
                self, "Nothing to save", "Run Compute group stats first before saving."
            )
            return
        if self._bids_root is None:
            QMessageBox.warning(
                self,
                "No BIDS root",
                "No BIDS root was provided when the viewer was launched. "
                "Cannot determine where to write output files.",
            )
            return

        if self._analysis_mode == "slope":
            default_pipeline_label = "trial_slope_stats_group"
            default_output_description = "trialslopestatsgroup"
        else:
            default_pipeline_label = "trial_stats_group"
            default_output_description = "trialstatsgroup"

        dlg = SaveTrialStatsGroupDialog(
            self._bids_root,
            parent=self,
            default_pipeline_label=default_pipeline_label,
            default_output_description=default_output_description,
        )
        if dlg.exec() != SaveTrialStatsGroupDialog.DialogCode.Accepted:
            return

        if self._analysis_mode == "slope":
            writer_params = TrialSlopeStatsGroupWriterParams(**dlg.get_common_writer_kwargs())
            writer = TrialSlopeStatsGroupProcessingWriter(writer_params)
        else:
            writer_params = dlg.get_writer_params()
            from gin_bids_py_analysis.processing.trial_stats_group import (
                TrialStatsGroupProcessingWriter,
            )
            writer = TrialStatsGroupProcessingWriter(writer_params)

        self._group_params_panel.set_save_enabled(False)
        self._group_params_panel.set_status("Saving group result…")

        if self._write_group_worker is not None:
            try:
                self._write_group_worker.finished.disconnect()
                self._write_group_worker.error.disconnect()
            except RuntimeError:
                pass

        worker = WriteGroupWorker(self._group_result, writer, parent=self)
        self._write_group_worker = worker
        worker.finished.connect(self._on_write_group_finished)
        worker.error.connect(self._on_write_group_error)
        worker.start()

    def _on_write_group_finished(self) -> None:
        self._group_params_panel.set_status("Group result saved")
        self._group_params_panel.set_save_enabled(True)

    def _on_write_group_error(self, message: str) -> None:
        self._group_params_panel.set_save_enabled(True)
        self._group_params_panel.set_status(f"Save error: {message}")
        QMessageBox.critical(self, "Save error", message)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_placeholder_group_params() -> "TrialStatsGroupParams":
    """Return a minimal valid TrialStatsGroupParams for the disabled Group tab."""
    from gin_bids_py_analysis.processing.trial_stats_group.params import (
        TrialStatsGroupParams,
    )

    return TrialStatsGroupParams(
        roi_mode="manual",
        manual_region_channels={"placeholder": {"01": ["CH1"]}},
    )


def _make_placeholder_slope_group_params() -> TrialSlopeStatsGroupParams:
    return TrialSlopeStatsGroupParams(
        roi_mode="manual",
        manual_region_channels={"placeholder": {"01": ["CH1"]}},
    )


def _coerce_slope_group_params_from_ttest(
    params: TrialStatsGroupParams | TrialSlopeStatsGroupParams,
) -> TrialSlopeStatsGroupParams:
    if isinstance(params, TrialSlopeStatsGroupParams):
        return params

    correction_method = params.p_value_correction_method
    if correction_method == "cluster_permutation":
        correction_method = "none"

    return TrialSlopeStatsGroupParams(
        p_value_correction_method=correction_method,
        significance_alpha=params.significance_alpha,
        roi_mode=params.roi_mode,
        atlas_name=params.atlas_name,
        manual_region_channels=params.manual_region_channels,
        min_channels_per_roi=params.min_channels_per_roi,
        min_subjects_per_roi=params.min_subjects_per_roi,
    )

"""Main application window for trial statistics visualization."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QMainWindow, QSplitter, QWidget

from gin_bids_py_analysis.bids import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_stats import (
    TrialLabelResolver,
    TrialStatsParams,
    TrialStatsProcessing,
    TrialStatsProcessingResult,
)

from .panels.params_panel import ParamsPanel
from .panels.plot_panel import PlotPanel
from .panels.subject_panel import SubjectChannelPanel
from .worker import ComputeWorker, PreloadWorker


class TrialStatsWindow(QMainWindow):
    """Three-panel window for interactive trial statistics visualization.

    Layout (left → middle → right)
    --------------------------------
    ``SubjectChannelPanel`` | ``PlotPanel`` | ``ParamsPanel``

    Parameters
    ----------
    subject_groups:
        Mapping of subject id to ``BIDSFileGroup`` for that subject.
    default_params:
        Initial ``TrialStatsParams`` to pre-populate the right panel.
    resolver:
        The ``TrialLabelResolver`` configured for this dataset.
    """

    def __init__(
        self,
        subject_groups: dict[str, BIDSFileGroup],
        default_params: TrialStatsParams,
        resolver: TrialLabelResolver,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Trial Statistics Viewer")
        self.resize(1280, 760)

        self._subject_groups = subject_groups
        self._resolver = resolver
        self._current_result: TrialStatsProcessingResult | None = None
        self._current_worker: ComputeWorker | None = None
        # Guard variable: only accept results whose generation matches this value
        self._compute_generation: int = 0
        self._preload_worker: PreloadWorker | None = None

        # ------------------------------------------------------------------
        # Build panels
        # ------------------------------------------------------------------
        subject_ids = sorted(subject_groups.keys())
        self._subject_panel = SubjectChannelPanel(subject_ids)
        self._plot_panel = PlotPanel()
        self._params_panel = ParamsPanel(default_params)

        # ------------------------------------------------------------------
        # Splitter layout
        # ------------------------------------------------------------------
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._subject_panel)
        splitter.addWidget(self._plot_panel)
        splitter.addWidget(self._params_panel)
        splitter.setSizes([200, 740, 340])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        self.setCentralWidget(splitter)

        # ------------------------------------------------------------------
        # Wire signals
        # ------------------------------------------------------------------
        self._subject_panel.subject_changed.connect(self._on_subject_changed)
        self._subject_panel.channel_changed.connect(self._on_channel_changed)
        self._params_panel.compute_requested.connect(self._on_compute_requested)

        # Pre-load all files in the background; auto-compute first subject once done
        if subject_ids:
            QTimer.singleShot(0, self._start_preload)

    # ------------------------------------------------------------------
    # Slot handlers
    # ------------------------------------------------------------------

    def _trigger_initial_compute(self) -> None:
        self._start_compute(self._subject_panel.current_subject)

    def _on_subject_changed(self, subject_id: str) -> None:
        try:
            params = self._params_panel.get_params()
        except ValueError:
            return
        self._start_compute(subject_id, params=params)

    def _on_channel_changed(self, channel_idx: int) -> None:
        if self._current_result is not None and channel_idx >= 0:
            self._plot_panel.update_plots(self._current_result, channel_idx)

    def _on_compute_requested(self) -> None:
        try:
            params = self._params_panel.get_params()
        except ValueError:
            return
        self._start_compute(self._subject_panel.current_subject, params=params)

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
    # Computation lifecycle
    # ------------------------------------------------------------------

    def _start_compute(
        self,
        subject_id: str,
        params: TrialStatsParams | None = None,
    ) -> None:
        group = self._subject_groups.get(subject_id)
        if group is None:
            return

        if params is None:
            try:
                params = self._params_panel.get_params()
            except ValueError:
                return

        # Bump generation so any in-flight result is discarded
        self._compute_generation += 1
        current_gen = self._compute_generation

        # Disconnect previous worker signals (let it finish silently)
        if self._current_worker is not None:
            try:
                self._current_worker.result_ready.disconnect()
                self._current_worker.error.disconnect()
            except RuntimeError:
                pass
            self._current_worker = None

        self._plot_panel.show_placeholder()
        self._params_panel.set_computing(True)

        processor = TrialStatsProcessing(params, resolver=self._resolver)
        worker = ComputeWorker(processor, group, parent=self)
        self._current_worker = worker

        worker.result_ready.connect(
            lambda result, gen=current_gen: self._on_result_ready(result, gen)
        )
        worker.error.connect(
            lambda msg, gen=current_gen: self._on_worker_error(msg, gen)
        )
        worker.start()

    def _on_result_ready(
        self,
        result: TrialStatsProcessingResult,
        generation: int,
    ) -> None:
        if generation != self._compute_generation:
            return  # stale result — discard

        self._current_result = result
        self._params_panel.set_computing(False)
        self._params_panel.set_status(
            f"Done — {result.condition_a_trial_count}× {result.condition_a} / "
            f"{result.condition_b_trial_count}× {result.condition_b}"
        )

        # Repopulate channel list, restoring the previously selected channel by name
        previous_channel = self._subject_panel.current_channel_name
        self._subject_panel.set_channels(result.channel_names, restore_name=previous_channel)
        # Plot the selected channel
        self._plot_panel.update_plots(result, self._subject_panel.current_channel_index)

    def _on_worker_error(self, message: str, generation: int) -> None:
        if generation != self._compute_generation:
            return  # stale

        self._params_panel.set_computing(False)
        self._params_panel.set_status(f"Error: {message}")

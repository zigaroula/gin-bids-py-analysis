"""Visualization window for pre-computed trial statistics.

This window loads already-computed ``.h5`` / ``.mat`` stat files from disk
instead of running the full processing pipeline.  It is the entry point for
the *precomputed* visualization path — useful when the per-subject pipeline was
run with slow methods (e.g. permutation tests) and the results are already on
disk.

Subject results are loaded once on startup.  The Group tab behaves identically
to the standard :class:`TrialStatsWindow`: group stats can be either loaded
directly from a pre-computed group file *or* (re-)computed on demand from the
loaded subject results using a :class:`GroupParamsPanel`.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from gin_bids_py_analysis.processing.trial_stats import (
    ConditionTestProcessingResult,
)
from gin_bids_py_analysis.processing.trial_stats import (
    RegressionProcessingResult,
)
from gin_bids_py_analysis.processing.trial_stats_group.processor import (
    format_manual_roi_missing_channels_message,
)

from .panels.group_params_panel import GroupParamsPanel
from .panels.group_plot_panel import GroupPlotPanel
from .panels.plot_panel import PlotPanel
from .panels.subject_panel import SubjectChannelPanel
from .worker import (
    GroupComputeWorker,
    LoadGroupResultWorker,
    LoadSubjectResultsWorker,
)

if TYPE_CHECKING:
    from gin_bids_py_analysis.processing.trial_stats_group import (
        ConditionTestGroupParams,
    )
    from gin_bids_py_analysis.processing.trial_stats_group import (
        ConditionTestGroupProcessingResult,
    )


def _make_placeholder_group_params() -> "ConditionTestGroupParams":
    from gin_bids_py_analysis.processing.trial_stats_group import (
        ConditionTestGroupParams,
    )

    return ConditionTestGroupParams(
        roi_mode="manual",
        manual_region_channels={"placeholder": {"01": ["CH1"]}},
    )


class TrialStatsPrecomputedWindow(QMainWindow):
    """Trial statistics viewer that loads pre-computed results from files.

    Subject tab (left → middle → right)
    ------------------------------------
    ``SubjectChannelPanel`` | ``PlotPanel`` | ``_LoadStatusPanel``

    Group tab (left/middle → right)
    --------------------------------
    ``GroupPlotPanel`` (with built-in ROI list) | ``GroupParamsPanel``

    On startup the window loads every subject file listed in *subject_files*
    in a background thread.  Once all subjects are loaded the Group tab is
    enabled (if *group_params* or *group_file* is provided).

    Parameters
    ----------
    subject_files:
        Mapping of ``subject_id → Path`` to the per-subject stats file.
    group_file:
        Optional path to a pre-computed group stats file.  When provided the
        group result is loaded from disk and the Group tab is shown immediately
        after all subjects are ready.  The ``GroupParamsPanel`` is still shown
        so the user can re-compute the group with different parameters.
    group_params:
        Optional ``ConditionTestGroupParams`` used to pre-populate the
        ``GroupParamsPanel``.  When *group_file* is ``None`` and *group_params*
        is set, the user can click *Compute group stats* to run the group
        analysis from the loaded subject results.
    parent:
        Optional Qt parent widget.
    """

    def __init__(
        self,
        subject_files: dict[str, Path],
        group_file: Path | None = None,
        group_params: "ConditionTestGroupParams | None" = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Trial Statistics Viewer (precomputed)")
        self.resize(1280, 760)

        self._subject_files = subject_files
        self._group_file = group_file
        self._group_params = group_params
        self._current_result: ConditionTestProcessingResult | RegressionProcessingResult | None = None
        self._all_results: dict[str, ConditionTestProcessingResult | RegressionProcessingResult] = {}
        self._group_result: "ConditionTestGroupProcessingResult | None" = None
        self._load_worker: LoadSubjectResultsWorker | None = None
        self._load_group_worker: LoadGroupResultWorker | None = None
        self._group_worker: GroupComputeWorker | None = None

        subject_ids = sorted(subject_files.keys())

        # ------------------------------------------------------------------
        # Subject tab panels
        # ------------------------------------------------------------------
        self._subject_panel = SubjectChannelPanel(subject_ids)
        self._plot_panel = PlotPanel()
        self._status_panel = _LoadStatusPanel()

        subject_splitter = QSplitter(Qt.Orientation.Horizontal)
        subject_splitter.addWidget(self._subject_panel)
        subject_splitter.addWidget(self._plot_panel)
        subject_splitter.addWidget(self._status_panel)
        subject_splitter.setSizes([200, 820, 260])
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
        if group_params is not None:
            self._group_params_panel = GroupParamsPanel(
                group_params, subject_ids=subject_ids
            )
        else:
            self._group_params_panel = GroupParamsPanel(
                _make_placeholder_group_params(), subject_ids=subject_ids
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
        self._tabs.setTabEnabled(1, False)

        # ------------------------------------------------------------------
        # Wire signals
        # ------------------------------------------------------------------
        self._subject_panel.subject_changed.connect(self._on_subject_changed)
        self._subject_panel.channel_changed.connect(self._on_channel_changed)
        self._group_params_panel.compute_requested.connect(
            self._on_group_compute_requested
        )

        if subject_ids:
            QTimer.singleShot(0, self._start_load)

    # ------------------------------------------------------------------
    # Subject load lifecycle
    # ------------------------------------------------------------------

    def _start_load(self) -> None:
        n = len(self._subject_files)
        self._subject_panel.set_interactive(False)
        self._status_panel.set_status(f"Loading {n} subject file(s)…")

        worker = LoadSubjectResultsWorker(self._subject_files, parent=self)
        self._load_worker = worker
        worker.subject_done.connect(self._on_subject_loaded)
        worker.all_done.connect(self._on_all_loaded)
        worker.progress.connect(self._status_panel.set_status)
        worker.error.connect(self._on_load_error)
        worker.start()

    def _on_subject_loaded(
        self, subject_id: str, result: ConditionTestProcessingResult
    ) -> None:
        self._all_results[subject_id] = result
        if subject_id == self._subject_panel.current_subject:
            self._display_subject_result(subject_id, result)

    def _on_all_loaded(
        self, results: dict[str, ConditionTestProcessingResult | RegressionProcessingResult]
    ) -> None:
        self._subject_panel.set_interactive(True)
        n = len(results)
        self._status_panel.set_status(f"Loaded {n} subject(s)")

        current = self._subject_panel.current_subject
        if current in results and self._current_result is not results.get(current):
            self._display_subject_result(current, results[current])

        has_slope = any(_is_slope_result(result) for result in results.values())
        # Enable Group tab when group_params or group_file provided (ttest only).
        if has_slope:
            self._tabs.setTabEnabled(1, False)
            self._group_params_panel.set_status(
                "Group tab disabled for slope precomputed files (V1)."
            )
            return
        if self._group_params is not None or self._group_file is not None:
            self._tabs.setTabEnabled(1, True)
            if self._group_file is not None:
                self._group_params_panel.set_status("Loading group stats from file…")
                QTimer.singleShot(0, self._start_load_group)
            else:
                self._group_params_panel.set_status(
                    f"{n} subject(s) ready — click 'Compute group stats'"
                )

    def _on_load_error(self, message: str) -> None:
        self._subject_panel.set_interactive(True)
        self._status_panel.set_status(f"Load error: {message}")

    def _display_subject_result(
        self,
        subject_id: str,
        result: ConditionTestProcessingResult | RegressionProcessingResult,
    ) -> None:
        self._current_result = result
        n_ok = len(self._all_results)
        n_total = len(self._subject_files)
        self._status_panel.set_status(
            f"[{n_ok}/{n_total}] {subject_id} — "
            f"{result.condition_a_trial_count}× {result.condition_a} / "
            f"{result.condition_b_trial_count}× {result.condition_b}"
        )
        previous = self._subject_panel.current_channel_name
        self._subject_panel.set_channels(
            result.channel_names, restore_name=previous
        )
        self._plot_panel.update_plots(result, self._subject_panel.current_channel_index)

    # ------------------------------------------------------------------
    # Subject tab slots
    # ------------------------------------------------------------------

    def _on_subject_changed(self, subject_id: str) -> None:
        if subject_id in self._all_results:
            self._display_subject_result(subject_id, self._all_results[subject_id])

    def _on_channel_changed(self, channel_idx: int) -> None:
        if self._current_result is not None and channel_idx >= 0:
            self._plot_panel.update_plots(self._current_result, channel_idx)

    # ------------------------------------------------------------------
    # Group: load from file lifecycle
    # ------------------------------------------------------------------

    def _start_load_group(self) -> None:
        assert self._group_file is not None
        worker = LoadGroupResultWorker(self._group_file, parent=self)
        self._load_group_worker = worker
        worker.result_ready.connect(self._on_group_loaded)
        worker.error.connect(self._on_group_load_error)
        worker.start()

    def _on_group_loaded(
        self, result: "ConditionTestGroupProcessingResult"
    ) -> None:
        self._group_result = result
        n_rois = len(result.region_names)
        excluded = len(result.excluded_rois)
        status = f"Loaded — {n_rois} ROI(s)"
        if excluded:
            status += f" ({excluded} excluded)"
        self._group_params_panel.set_status(status)
        self._group_plot_panel.update_plots(result, 0)

    def _on_group_load_error(self, message: str) -> None:
        self._group_params_panel.set_status(f"Load error: {message}")

    # ------------------------------------------------------------------
    # Group: compute from loaded subjects lifecycle
    # ------------------------------------------------------------------

    def _on_group_compute_requested(self) -> None:
        if any(_is_slope_result(result) for result in self._all_results.values()):
            self._group_params_panel.set_status(
                "Group statistics are unavailable in slope mode (V1)."
            )
            return
        if not self._all_results:
            self._group_params_panel.set_status("No subject results loaded yet.")
            return
        try:
            params = self._group_params_panel.get_params()
        except ValueError:
            return
        self._start_group_compute(params)

    def _start_group_compute(self, params: "ConditionTestGroupParams") -> None:
        if self._group_worker is not None:
            try:
                self._group_worker.result_ready.disconnect()
                self._group_worker.error.disconnect()
            except RuntimeError:
                pass
            self._group_worker = None

        self._group_plot_panel.show_placeholder()
        self._group_params_panel.set_computing(True)
        self._group_params_panel.set_log_message("")
        self._group_params_panel.set_status(
            f"Computing group stats for {len(self._all_results)} subject(s)\u2026"
        )

        worker = GroupComputeWorker(self._all_results, params, parent=self)
        self._group_worker = worker
        worker.result_ready.connect(self._on_group_compute_done)
        worker.error.connect(self._on_group_compute_error)
        worker.start()

    def _on_group_compute_done(
        self, result: "ConditionTestGroupProcessingResult"
    ) -> None:
        self._group_result = result
        self._group_params_panel.set_computing(False)
        n_rois = len(result.region_names)
        excluded = len(result.excluded_rois)
        status = f"Done \u2014 {n_rois} ROI(s)"
        if excluded:
            status += f" ({excluded} excluded)"
        self._group_params_panel.set_status(status)
        self._group_params_panel.set_log_message(
            _group_result_log_message(result)
        )
        self._group_plot_panel.update_plots(result, 0)

    def _on_group_compute_error(self, message: str) -> None:
        self._group_params_panel.set_computing(False)
        self._group_params_panel.set_log_message("")
        self._group_params_panel.set_status(f"Error: {message}")


# ---------------------------------------------------------------------------
# Minimal status panel (right side of Subject tab)
# ---------------------------------------------------------------------------


class _LoadStatusPanel(QWidget):
    """Minimal right-side panel showing the loading / subject status.

    Unlike :class:`ParamsPanel` this panel has no compute button — subject
    results are loaded from files and cannot be re-parameterized in-place.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(180)
        self.setMaximumWidth(300)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        self._status_label = QLabel("Loading…")
        self._status_label.setWordWrap(True)
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._status_label)
        layout.addStretch()

    def set_status(self, message: str) -> None:
        """Update the status text."""
        self._status_label.setText(message)


def _is_slope_result(result: object) -> bool:
    return hasattr(result, "analysis_type") and getattr(result, "analysis_type", "") == "slope_regression"


def _group_result_log_message(result: object) -> str:
    missing_manual_channels = getattr(result, "manual_roi_missing_channels", {})
    if not missing_manual_channels:
        return ""
    return format_manual_roi_missing_channels_message(missing_manual_channels)



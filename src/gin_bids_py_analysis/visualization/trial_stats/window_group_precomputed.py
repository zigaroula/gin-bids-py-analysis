"""Visualization window for a pre-computed group trial statistics file.

This window loads a single ``TrialStatsGroupProcessingResult`` from disk and
displays the group-level ROI plots without requiring per-subject files.  It is
the entry point for the *group-only precomputed* visualization path — useful
when group stats have already been written by
``TrialStatsGroupProcessingWriter`` and you only want to inspect or re-export
the group results.

Layout
------
``GroupPlotPanel`` (left) | ``GroupParamsPanel`` (right, read-only status)

The ``GroupParamsPanel`` shows the status of the load operation.  Its
*Compute group stats* button is intentionally non-functional in this window
because no per-subject results are available.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QMainWindow,
    QSplitter,
    QWidget,
)

from .panels.group_params_panel import GroupParamsPanel
from .panels.group_plot_panel import GroupPlotPanel
from .worker import LoadGroupResultWorker

if TYPE_CHECKING:
    from gin_bids_py_analysis.processing.trial_stats_group.params import (
        TrialStatsGroupParams,
    )
    from gin_bids_py_analysis.processing.trial_stats_group.result import (
        TrialStatsGroupProcessingResult,
    )


def _make_placeholder_group_params() -> "TrialStatsGroupParams":
    from gin_bids_py_analysis.processing.trial_stats_group.params import (
        TrialStatsGroupParams,
    )

    return TrialStatsGroupParams(
        roi_mode="manual",
        manual_region_channels={"placeholder": {"01": ["CH1"]}},
    )


class TrialStatsGroupPrecomputedWindow(QMainWindow):
    """Group trial statistics viewer that loads a pre-computed group file.

    Parameters
    ----------
    group_file:
        Path to the pre-computed group stats file written by
        ``TrialStatsGroupProcessingWriter`` (``.h5``/``.hdf5``).
    group_params:
        Optional ``TrialStatsGroupParams`` used only to pre-populate the
        ``GroupParamsPanel`` display.  No computation is performed from this
        window.
    parent:
        Optional Qt parent widget.
    """

    def __init__(
        self,
        group_file: Path,
        group_params: "TrialStatsGroupParams | None" = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Trial Statistics Group Viewer (precomputed)")
        self.resize(1280, 760)

        self._group_file = group_file
        self._group_result: "TrialStatsGroupProcessingResult | None" = None
        self._load_worker: LoadGroupResultWorker | None = None

        # ------------------------------------------------------------------
        # Panels
        # ------------------------------------------------------------------
        self._group_plot_panel = GroupPlotPanel()
        if group_params is not None:
            self._group_params_panel = GroupParamsPanel(group_params, subject_ids=[])
        else:
            self._group_params_panel = GroupParamsPanel(
                _make_placeholder_group_params(), subject_ids=[]
            )

        # Disable compute button — no subject results are available
        self._group_params_panel.set_computing(False)
        self._group_params_panel.set_status("Loading group stats from file…")

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._group_plot_panel)
        splitter.addWidget(self._group_params_panel)
        splitter.setSizes([940, 340])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)
        self.setCentralWidget(central)

        # Wire compute button to an informative no-op
        self._group_params_panel.compute_requested.connect(
            self._on_compute_requested
        )

        QTimer.singleShot(0, self._start_load)

    # ------------------------------------------------------------------
    # Load lifecycle
    # ------------------------------------------------------------------

    def _start_load(self) -> None:
        worker = LoadGroupResultWorker(self._group_file, parent=self)
        self._load_worker = worker
        worker.result_ready.connect(self._on_group_loaded)
        worker.error.connect(self._on_load_error)
        worker.start()

    def _on_group_loaded(
        self, result: "TrialStatsGroupProcessingResult"
    ) -> None:
        self._group_result = result
        n_rois = len(result.region_names)
        excluded = len(result.excluded_rois)
        status = f"Loaded — {n_rois} ROI(s)"
        if excluded:
            status += f" ({excluded} excluded)"
        self._group_params_panel.set_status(status)
        self._group_plot_panel.update_plots(result, 0)

    def _on_load_error(self, message: str) -> None:
        self._group_params_panel.set_status(f"Load error: {message}")

    def _on_compute_requested(self) -> None:
        self._group_params_panel.set_status(
            "Re-computation requires per-subject results. "
            "Use 'visualize_trial_stats_precomputed.py' instead."
        )

"""Right panel for group-level analysis: TrialStatsGroupParams form + Compute button."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from pydantic import ValidationError

from gin_bids_py_analysis.processing.trial_stats_group.params import (
    TrialStatsGroupParams,
)
from .manual_region_channels_dialog import ManualRegionChannelsDialog


class GroupParamsPanel(QWidget):
    """Right panel with ``TrialStatsGroupParams`` fields and a Compute button.

    The ``manual_region_channels`` field is edited via a dedicated dialog
    opened with the **Edit regions…** button.

    Signals
    -------
    compute_requested : emitted when the user clicks *Compute group stats*.
    save_requested    : emitted when the user clicks *Save results*.
    """

    compute_requested = Signal()
    save_requested = Signal()

    def __init__(
        self,
        params: TrialStatsGroupParams,
        subject_ids: list[str] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._subject_ids: list[str] = list(subject_ids) if subject_ids else []
        self.setMinimumWidth(280)
        self.setMaximumWidth(400)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Scrollable form
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setContentsMargins(8, 8, 8, 8)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        # source_metric
        self._source_metric = QComboBox()
        for m in ("mean_difference", "t_values", "condition_a_mean", "condition_b_mean"):
            self._source_metric.addItem(m)
        form.addRow("Source metric", self._source_metric)

        # p_value_correction_method
        self._correction = QComboBox()
        for m in ("none", "fdr_bh", "bonferroni", "cluster_permutation"):
            self._correction.addItem(m)
        form.addRow("p-value correction", self._correction)

        # cluster_permutation_method
        self._cluster_method = QComboBox()
        for m in ("custom", "mne"):
            self._cluster_method.addItem(m)
        form.addRow("Cluster method", self._cluster_method)

        # significance_alpha
        self._alpha = QDoubleSpinBox()
        self._alpha.setRange(0.0001, 0.9999)
        self._alpha.setDecimals(4)
        self._alpha.setSingleStep(0.01)
        form.addRow("Significance α", self._alpha)

        # roi_mode (informational; changing it does not affect manual_region_channels UI)
        self._roi_mode = QComboBox()
        for m in ("manual", "atlas"):
            self._roi_mode.addItem(m)
        form.addRow("ROI mode", self._roi_mode)

        # atlas_name (only relevant when roi_mode="atlas")
        self._atlas_name = QLineEdit()
        self._atlas_name.setPlaceholderText("(unused for manual ROI mode)")
        form.addRow("Atlas name", self._atlas_name)

        # min_channels_per_roi
        self._min_channels = QSpinBox()
        self._min_channels.setRange(1, 9999)
        form.addRow("Min channels / ROI", self._min_channels)

        # min_subjects_per_roi
        self._min_subjects = QSpinBox()
        self._min_subjects.setRange(1, 9999)
        form.addRow("Min subjects / ROI", self._min_subjects)

        # manual_region_channels — button opens editor dialog
        self._edit_regions_btn = QPushButton("Edit regions…")
        self._edit_regions_btn.setToolTip(
            "Open a dialog to configure manual region-to-channel assignments"
        )
        form.addRow("Region channels", self._edit_regions_btn)
        self._regions_summary = QLabel()
        self._regions_summary.setStyleSheet("color: gray; font-size: 11px;")
        self._regions_summary.setWordWrap(True)
        form.addRow("", self._regions_summary)

        scroll.setWidget(form_widget)
        outer.addWidget(scroll, stretch=1)

        # Bottom: Compute + status
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(8, 4, 8, 8)

        self._compute_btn = QPushButton("Compute group stats")
        self._compute_btn.setFixedHeight(34)
        bottom_layout.addWidget(self._compute_btn)

        self._save_btn = QPushButton("Save results")
        self._save_btn.setFixedHeight(34)
        self._save_btn.setEnabled(False)
        bottom_layout.addWidget(self._save_btn)

        self._status_label = QLabel("Waiting for all subjects…")
        self._status_label.setWordWrap(True)
        bottom_layout.addWidget(self._status_label)

        outer.addWidget(bottom)

        self._compute_btn.clicked.connect(self.compute_requested)
        self._save_btn.clicked.connect(self.save_requested)
        self._edit_regions_btn.clicked.connect(self._open_regions_dialog)

        # Store initial params so get_params() can round-trip manual_region_channels
        self._manual_region_channels: dict = {}
        self.set_params(params)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_params(self) -> TrialStatsGroupParams:
        """Parse current widget values into a ``TrialStatsGroupParams``.

        Raises
        ------
        ValueError
            When Pydantic validation fails.  A ``QMessageBox`` is also shown.
        """
        atlas_name = self._atlas_name.text().strip() or None
        try:
            return TrialStatsGroupParams(
                source_metric=self._source_metric.currentText(),
                p_value_correction_method=self._correction.currentText(),
                cluster_permutation_method=self._cluster_method.currentText(),
                significance_alpha=self._alpha.value(),
                roi_mode=self._roi_mode.currentText(),
                atlas_name=atlas_name,
                manual_region_channels=self._manual_region_channels,
                min_channels_per_roi=self._min_channels.value(),
                min_subjects_per_roi=self._min_subjects.value(),
            )
        except (ValidationError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid parameters", str(exc))
            raise ValueError(str(exc)) from exc

    def set_params(self, params: TrialStatsGroupParams) -> None:
        """Populate all widgets from *params*."""
        _set_combo(self._source_metric, params.source_metric)
        _set_combo(self._correction, params.p_value_correction_method)
        _set_combo(self._cluster_method, params.cluster_permutation_method)
        self._alpha.setValue(params.significance_alpha)
        _set_combo(self._roi_mode, params.roi_mode)
        self._atlas_name.setText(params.atlas_name or "")
        self._min_channels.setValue(params.min_channels_per_roi)
        self._min_subjects.setValue(params.min_subjects_per_roi)
        self._manual_region_channels = dict(params.manual_region_channels)
        self._update_regions_summary()

    def set_computing(self, computing: bool) -> None:
        """Disable/enable the Compute button while a computation is running."""
        self._compute_btn.setEnabled(not computing)
        self._compute_btn.setText(
            "Computing…" if computing else "Compute group stats"
        )

    def set_save_enabled(self, enabled: bool) -> None:
        """Enable or disable the Save results button."""
        self._save_btn.setEnabled(enabled)

    def set_status(self, message: str) -> None:
        """Update the status label below the Compute button."""
        self._status_label.setText(message)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _open_regions_dialog(self) -> None:
        dlg = ManualRegionChannelsDialog(
            mapping=self._manual_region_channels,
            subject_ids=self._subject_ids,
            parent=self,
        )
        if dlg.exec() == ManualRegionChannelsDialog.DialogCode.Accepted:
            self._manual_region_channels = dlg.result_mapping()
            self._update_regions_summary()

    def _update_regions_summary(self) -> None:
        n_regions = len(self._manual_region_channels)
        if n_regions == 0:
            self._regions_summary.setText("No regions configured")
            return
        n_assignments = sum(
            len(subj_map)
            for subj_map in self._manual_region_channels.values()
        )
        region_word = "region" if n_regions == 1 else "regions"
        subj_word = "subject" if n_assignments == 1 else "subjects"
        self._regions_summary.setText(
            f"{n_regions} {region_word}, {n_assignments} {subj_word} with channels"
        )


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _set_combo(combo: QComboBox, value: str) -> None:
    idx = combo.findText(value)
    if idx >= 0:
        combo.setCurrentIndex(idx)

"""Right panel for group-level analysis: ConditionTestGroupParams form + Compute button."""

from __future__ import annotations

from typing import Literal

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

from gin_bids_py_analysis.processing.trial_stats_group import (
    RegressionGroupParams,
)
from gin_bids_py_analysis.processing.trial_stats_group import (
    ConditionTestGroupParams,
)
from .manual_region_channels_dialog import ManualRegionChannelsDialog


def _merge_model_params(
    model: ConditionTestGroupParams | RegressionGroupParams,
    **updates: object,
) -> ConditionTestGroupParams | RegressionGroupParams:
    """Return a validated copy of *model* with widget-driven updates applied."""
    payload = model.model_dump()
    payload.update(updates)
    return model.__class__(**payload)


class GroupParamsPanel(QWidget):
    """Right panel with ``ConditionTestGroupParams`` fields and a Compute button.

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
        params: ConditionTestGroupParams | RegressionGroupParams,
        subject_ids: list[str] | None = None,
        analysis_mode: Literal["ttest", "slope"] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._subject_ids: list[str] = list(subject_ids) if subject_ids else []
        self._analysis_mode: Literal["ttest", "slope"] = (
            analysis_mode
            if analysis_mode is not None
            else ("slope" if isinstance(params, RegressionGroupParams) else "ttest")
        )
        self._ttest_params: ConditionTestGroupParams | None = (
            params if isinstance(params, ConditionTestGroupParams) else None
        )
        self._slope_params: RegressionGroupParams | None = (
            params if isinstance(params, RegressionGroupParams) else None
        )
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

        # Primary group metric
        self._source_metric = QComboBox()
        for m in (
            "mean_difference",
            "t_values",
            "condition_a_mean",
            "condition_b_mean",
            "slope",
            "r_value",
        ):
            self._source_metric.addItem(m)
        self._source_metric_label = QLabel("Primary metric")
        form.addRow(self._source_metric_label, self._source_metric)

        self._contrast_mode = QComboBox()
        for m in ("paired", "unpaired"):
            self._contrast_mode.addItem(m)
        self._contrast_mode_label = QLabel("Contrast mode")
        form.addRow(self._contrast_mode_label, self._contrast_mode)

        # p_value_correction_method
        self._correction = QComboBox()
        correction_modes = (
            ("none", "fdr_bh", "bonferroni")
            if self._analysis_mode == "slope"
            else ("none", "fdr_bh", "bonferroni", "cluster_permutation")
        )
        for m in correction_modes:
            self._correction.addItem(m)
        form.addRow("p-value correction", self._correction)

        # cluster_permutation_method
        self._cluster_method = QComboBox()
        for m in ("custom", "mne"):
            self._cluster_method.addItem(m)
        self._cluster_method_label = QLabel("Cluster method")
        form.addRow(self._cluster_method_label, self._cluster_method)

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

        self._log_label = QLabel("")
        self._log_label.setWordWrap(True)
        self._log_label.setStyleSheet("color: #8a5a00; font-size: 11px;")
        self._log_label.setVisible(False)
        bottom_layout.addWidget(self._log_label)

        outer.addWidget(bottom)

        self._compute_btn.clicked.connect(self.compute_requested)
        self._save_btn.clicked.connect(self.save_requested)
        self._edit_regions_btn.clicked.connect(self._open_regions_dialog)

        # Store initial params so get_params() can round-trip manual_region_channels
        self._manual_region_channels: dict = {}
        self._apply_analysis_mode_visibility()
        self.set_params(params)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_params(self) -> ConditionTestGroupParams | RegressionGroupParams:
        """Parse current widget values into a group-params model.

        Raises
        ------
        ValueError
            When Pydantic validation fails.  A ``QMessageBox`` is also shown.
        """
        atlas_name = self._atlas_name.text().strip() or None
        try:
            if self._analysis_mode == "slope":
                base_params = self._slope_params or RegressionGroupParams(
                    roi_mode="manual",
                    manual_region_channels={"placeholder": {"01": ["CH1"]}},
                )
                params = _merge_model_params(
                    base_params,
                    primary_regression_metric=self._source_metric.currentText(),
                    contrast_mode=self._contrast_mode.currentText(),
                    p_value_correction_method=self._correction.currentText(),
                    significance_alpha=self._alpha.value(),
                    roi_mode=self._roi_mode.currentText(),
                    atlas_name=atlas_name,
                    manual_region_channels=self._manual_region_channels,
                    min_channels_per_roi=self._min_channels.value(),
                    min_subjects_per_roi=self._min_subjects.value(),
                )
                assert isinstance(params, RegressionGroupParams)
                self._slope_params = params
                return params
            base_params = self._ttest_params or ConditionTestGroupParams(
                roi_mode="manual",
                manual_region_channels={"placeholder": {"01": ["CH1"]}},
            )
            params = _merge_model_params(
                base_params,
                primary_condition_metric=self._source_metric.currentText(),
                p_value_correction_method=self._correction.currentText(),
                cluster_permutation_method=self._cluster_method.currentText(),
                significance_alpha=self._alpha.value(),
                roi_mode=self._roi_mode.currentText(),
                atlas_name=atlas_name,
                manual_region_channels=self._manual_region_channels,
                min_channels_per_roi=self._min_channels.value(),
                min_subjects_per_roi=self._min_subjects.value(),
            )
            assert isinstance(params, ConditionTestGroupParams)
            self._ttest_params = params
            return params
        except (ValidationError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid parameters", str(exc))
            raise ValueError(str(exc)) from exc

    def set_params(self, params: ConditionTestGroupParams | RegressionGroupParams) -> None:
        """Populate all widgets from *params*."""
        if isinstance(params, RegressionGroupParams):
            self._slope_params = params
        else:
            self._ttest_params = params
        if isinstance(params, RegressionGroupParams):
            primary_metric = getattr(params, "primary_regression_metric", "slope")
        else:
            primary_metric = getattr(params, "primary_condition_metric", "mean_difference")
        _set_combo(self._source_metric, primary_metric)
        _set_combo(self._contrast_mode, getattr(params, "contrast_mode", "paired"))
        _set_combo(self._correction, params.p_value_correction_method)
        _set_combo(self._cluster_method, getattr(params, "cluster_permutation_method", "custom"))
        self._alpha.setValue(params.significance_alpha)
        _set_combo(self._roi_mode, params.roi_mode)
        self._atlas_name.setText(params.atlas_name or "")
        self._min_channels.setValue(params.min_channels_per_roi)
        self._min_subjects.setValue(params.min_subjects_per_roi)
        self._manual_region_channels = dict(params.manual_region_channels)
        self._update_regions_summary()

    def set_analysis_mode(self, analysis_mode: Literal["ttest", "slope"]) -> None:
        """Switch panel behavior between ttest and slope group-params schemas."""
        if analysis_mode == self._analysis_mode:
            return
        self._analysis_mode = analysis_mode
        self._rebuild_correction_options()
        self._apply_analysis_mode_visibility()

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

    def set_log_message(self, message: str) -> None:
        """Update the optional log line shown below the status label."""
        text = message.strip()
        self._log_label.setText(text)
        self._log_label.setVisible(bool(text))

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

    def _rebuild_correction_options(self) -> None:
        current = self._correction.currentText().strip()
        self._correction.blockSignals(True)
        self._correction.clear()
        options = (
            ("none", "fdr_bh", "bonferroni")
            if self._analysis_mode == "slope"
            else ("none", "fdr_bh", "bonferroni", "cluster_permutation")
        )
        for option in options:
            self._correction.addItem(option)
        _set_combo(self._correction, current if current else "none")
        self._correction.blockSignals(False)

    def _apply_analysis_mode_visibility(self) -> None:
        is_ttest = self._analysis_mode == "ttest"
        self._source_metric_label.setVisible(True)
        self._source_metric.setVisible(True)
        self._source_metric_label.setText(
            "Primary condition metric" if is_ttest else "Regression metric"
        )
        self._contrast_mode_label.setVisible(not is_ttest)
        self._contrast_mode.setVisible(not is_ttest)
        self._cluster_method_label.setVisible(is_ttest)
        self._cluster_method.setVisible(is_ttest)
        self._refresh_source_metric_options()

    def _refresh_source_metric_options(self) -> None:
        is_ttest = self._analysis_mode == "ttest"
        current = self._source_metric.currentText().strip()
        self._source_metric.blockSignals(True)
        self._source_metric.clear()
        options = (
            ("mean_difference", "t_values", "condition_a_mean", "condition_b_mean")
            if is_ttest
            else (
                "slope",
                "r_value",
            )
        )
        for option in options:
            self._source_metric.addItem(option)
        default = "mean_difference" if is_ttest else "slope"
        _set_combo(self._source_metric, current if current in options else default)
        self._source_metric.blockSignals(False)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _set_combo(combo: QComboBox, value: str) -> None:
    idx = combo.findText(value)
    if idx >= 0:
        combo.setCurrentIndex(idx)



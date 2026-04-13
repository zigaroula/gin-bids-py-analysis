"""Right panel: ConditionTestParams form + Compute button."""

from __future__ import annotations

import json

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
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

from gin_bids_py_analysis.processing.trial_stats import RegressionParams
from gin_bids_py_analysis.processing.trial_stats import ConditionTestParams


def _merge_model_params(model: ConditionTestParams | RegressionParams, **updates: object) -> ConditionTestParams | RegressionParams:
    """Return a validated copy of *model* with widget-driven updates applied."""
    payload = model.model_dump()
    payload.update(updates)
    return model.__class__(**payload)


def _predictor_transform_to_json(value: object) -> str:
    if not value:
        return ""
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if hasattr(item, "model_dump"):
                normalized[str(key)] = item.model_dump()
            else:
                normalized[str(key)] = item
        return json.dumps(
            normalized,
            sort_keys=True,
            ensure_ascii=True,
            separators=(",", ":"),
        )
    return ""


class ParamsPanel(QWidget):
    """Right panel with all ``ConditionTestParams`` fields and a Compute button.

    Signals
    -------
    compute_requested : emitted when the user clicks *Compute*.
    save_requested    : emitted when the user clicks *Save results*.
    """

    compute_requested = Signal()
    save_requested = Signal()

    def __init__(
        self,
        params: ConditionTestParams,
        slope_params: RegressionParams | None = None,
        default_mode: str = "ttest",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.setMaximumWidth(400)
        self._ttest_params = params
        self._slope_params = slope_params or RegressionParams(
            anchor_event_codes=list(params.anchor_event_codes),
            tmin_s=params.tmin_s,
            tmax_s=params.tmax_s,
            condition_a=params.condition_a,
            condition_b=params.condition_b,
            min_trials_per_condition=max(3, params.min_trials_per_condition),
            drop_partial_epochs=params.drop_partial_epochs,
            atlas_name=params.atlas_name,
            atlas_regions=list(params.atlas_regions),
            window_ms=params.window_ms,
            n_bins=params.n_bins,
            activity_zscore=params.activity_zscore,
            activity_baseline_tmin_s=params.activity_baseline_tmin_s,
            activity_baseline_tmax_s=params.activity_baseline_tmax_s,
            activity_baseline_scope=params.activity_baseline_scope,
            activity_baseline_remove_outlier_trial_means=params.activity_baseline_remove_outlier_trial_means,
        )

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        # Scrollable form area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.setContentsMargins(8, 8, 8, 8)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self._form = form

        # analysis mode
        self._analysis_mode = QComboBox()
        self._analysis_mode.addItem("ttest")
        self._analysis_mode.addItem("slope")
        form.addRow("Analysis mode", self._analysis_mode)

        # anchor_event_codes
        self._anchor_codes = QLineEdit()
        self._anchor_codes.setPlaceholderText("10, 20")
        form.addRow("Anchor event codes", self._anchor_codes)

        # tmin_s / tmax_s
        self._tmin = QDoubleSpinBox()
        self._tmin.setRange(-3600.0, 3600.0)
        self._tmin.setDecimals(3)
        self._tmin.setSingleStep(0.5)
        form.addRow("t min (s)", self._tmin)

        self._tmax = QDoubleSpinBox()
        self._tmax.setRange(-3600.0, 3600.0)
        self._tmax.setDecimals(3)
        self._tmax.setSingleStep(0.5)
        form.addRow("t max (s)", self._tmax)

        # condition_a / condition_b
        self._condition_a = QLineEdit()
        form.addRow("Condition A", self._condition_a)

        self._condition_b = QLineEdit()
        form.addRow("Condition B", self._condition_b)

        # slope-specific predictor settings
        self._predictor = QLineEdit()
        self._predictor.setPlaceholderText("predictor_value")
        form.addRow("Predictor", self._predictor)

        self._predictor_transform = QLineEdit()
        self._predictor_transform.setPlaceholderText(
            '{"pleasant":{"scale":1,"offset":0},"unpleasant":{"scale":-1,"offset":0}}'
        )
        form.addRow("Predictor transform", self._predictor_transform)

        self._predictor_scaling = QComboBox()
        self._predictor_scaling.addItem("none")
        self._predictor_scaling.addItem("condition")
        self._predictor_scaling.addItem("global")
        form.addRow("Predictor z-score", self._predictor_scaling)

        self._trial_activity_summary_kind = QComboBox()
        self._trial_activity_summary_kind.addItem("epoch_mean")
        self._trial_activity_summary_kind.addItem("anchor_to_response_mean")
        form.addRow("Trial activity summary", self._trial_activity_summary_kind)

        self._trial_activity_summary_source = QComboBox()
        self._trial_activity_summary_source.addItem("table_column")
        self._trial_activity_summary_source.addItem("annotation_event_code")
        form.addRow("Response source", self._trial_activity_summary_source)

        self._trial_activity_summary_response_column = QLineEdit()
        self._trial_activity_summary_response_column.setPlaceholderText("response_time")
        form.addRow("Response column", self._trial_activity_summary_response_column)

        self._trial_activity_summary_response_units = QComboBox()
        self._trial_activity_summary_response_units.addItem("s")
        self._trial_activity_summary_response_units.addItem("ms")
        form.addRow("Response units", self._trial_activity_summary_response_units)

        self._trial_activity_summary_response_event_code = QLineEdit()
        self._trial_activity_summary_response_event_code.setPlaceholderText("20")
        form.addRow("Response event code", self._trial_activity_summary_response_event_code)

        # min_trials_per_condition
        self._min_trials = QSpinBox()
        self._min_trials.setRange(2, 9999)
        form.addRow("Min trials / condition", self._min_trials)

        # drop_partial_epochs
        self._drop_partial = QCheckBox()
        form.addRow("Drop partial epochs", self._drop_partial)

        # equal_var
        self._equal_var = QCheckBox()
        self._equal_var.setToolTip("Check for Student's t-test; uncheck for Welch's (recommended)")
        form.addRow("Equal variance (Student)", self._equal_var)

        # p_value_correction_method
        self._correction = QComboBox()
        for method in ("none", "fdr_bh", "bonferroni", "permutation"):
            self._correction.addItem(method)
        form.addRow("p-value correction", self._correction)

        # significance_alpha
        self._alpha = QDoubleSpinBox()
        self._alpha.setRange(0.0001, 0.9999)
        self._alpha.setDecimals(4)
        self._alpha.setSingleStep(0.01)
        form.addRow("Significance α", self._alpha)

        # atlas_name
        self._atlas_name = QLineEdit()
        self._atlas_name.setPlaceholderText("(none — channel-level stats)")
        form.addRow("Atlas name", self._atlas_name)

        # atlas_regions
        self._atlas_regions = QLineEdit()
        self._atlas_regions.setPlaceholderText("(all regions)")
        form.addRow("Atlas regions", self._atlas_regions)

        # window_ms
        self._window_ms = QDoubleSpinBox()
        self._window_ms.setRange(0.0, 100_000.0)
        self._window_ms.setDecimals(1)
        self._window_ms.setSingleStep(10.0)
        self._window_ms.setSpecialValueText("disabled")
        form.addRow("Window (ms)", self._window_ms)

        # n_bins
        self._n_bins = QSpinBox()
        self._n_bins.setRange(0, 10_000)
        self._n_bins.setSpecialValueText("disabled")
        form.addRow("n bins", self._n_bins)

        self._activity_scaling = QComboBox()
        self._activity_scaling.addItem("none")
        self._activity_scaling.addItem("baseline")
        form.addRow("Activity z-score", self._activity_scaling)

        self._activity_baseline_tmin = QDoubleSpinBox()
        self._activity_baseline_tmin.setRange(-3600.0, 3600.0)
        self._activity_baseline_tmin.setDecimals(3)
        self._activity_baseline_tmin.setSingleStep(0.1)
        form.addRow("Activity baseline t min (s)", self._activity_baseline_tmin)

        self._activity_baseline_tmax = QDoubleSpinBox()
        self._activity_baseline_tmax.setRange(-3600.0, 3600.0)
        self._activity_baseline_tmax.setDecimals(3)
        self._activity_baseline_tmax.setSingleStep(0.1)
        form.addRow("Activity baseline t max (s)", self._activity_baseline_tmax)

        self._activity_baseline_scope = QComboBox()
        self._activity_baseline_scope.addItem("trial")
        self._activity_baseline_scope.addItem("condition")
        self._activity_baseline_scope.addItem("global")
        form.addRow("Activity baseline scope", self._activity_baseline_scope)

        self._activity_baseline_remove_outliers = QCheckBox()
        self._activity_baseline_remove_outliers.setToolTip(
            "Remove outlier baseline trial-means before estimating the baseline reference."
        )
        form.addRow(
            "Remove outlier trial means",
            self._activity_baseline_remove_outliers,
        )

        # Mutual exclusion: setting one to nonzero zeros the other
        self._window_ms.valueChanged.connect(self._on_window_ms_changed)
        self._n_bins.valueChanged.connect(self._on_n_bins_changed)

        # ---- channel significance ----
        _sig_sep = QLabel("— Channel significance —")
        _sig_sep.setStyleSheet("color: gray; font-size: 10px;")
        form.addRow(_sig_sep)

        # channel_significance_mode
        self._sig_mode = QComboBox()
        for _mode in ("none", "single_bin", "duration"):
            self._sig_mode.addItem(_mode)
        form.addRow("Mode", self._sig_mode)

        # channel_significance_duration_threshold_ms
        self._sig_duration_threshold = QDoubleSpinBox()
        self._sig_duration_threshold.setRange(0.1, 100_000.0)
        self._sig_duration_threshold.setDecimals(1)
        self._sig_duration_threshold.setSingleStep(10.0)
        self._sig_duration_threshold.setSuffix(" ms")
        form.addRow("Duration threshold", self._sig_duration_threshold)

        self._sig_mode.currentTextChanged.connect(self._on_sig_mode_changed)
        self._analysis_mode.currentTextChanged.connect(self._on_analysis_mode_changed)
        self._activity_scaling.currentTextChanged.connect(self._on_activity_scaling_changed)
        self._trial_activity_summary_kind.currentTextChanged.connect(
            self._on_trial_activity_summary_kind_changed
        )
        self._trial_activity_summary_source.currentTextChanged.connect(
            self._on_trial_activity_summary_source_changed
        )

        scroll.setWidget(form_widget)
        outer.addWidget(scroll, stretch=1)

        # Bottom: Compute button + status label
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(8, 4, 8, 8)

        self._compute_btn = QPushButton("Compute")
        self._compute_btn.setFixedHeight(34)
        bottom_layout.addWidget(self._compute_btn)

        self._save_btn = QPushButton("Save results")
        self._save_btn.setFixedHeight(34)
        self._save_btn.setEnabled(False)
        bottom_layout.addWidget(self._save_btn)

        self._status_label = QLabel("Ready")
        self._status_label.setWordWrap(True)
        bottom_layout.addWidget(self._status_label)

        outer.addWidget(bottom)

        self._compute_btn.clicked.connect(self.compute_requested)
        self._save_btn.clicked.connect(self.save_requested)

        # Populate initial values
        self.set_params(params)
        self._predictor.setText(self._slope_params.predictor)
        self._predictor_transform.setText(
            _predictor_transform_to_json(
                self._slope_params.predictor_transform_by_condition
            )
        )
        idx_scale = self._predictor_scaling.findText(self._slope_params.predictor_zscore)
        if idx_scale >= 0:
            self._predictor_scaling.setCurrentIndex(idx_scale)
        idx_baseline_scope = self._activity_baseline_scope.findText(
            self._slope_params.activity_baseline_scope
        )
        if idx_baseline_scope >= 0:
            self._activity_baseline_scope.setCurrentIndex(idx_baseline_scope)
        self._activity_baseline_remove_outliers.setChecked(
            self._slope_params.activity_baseline_remove_outlier_trial_means
        )
        self._set_trial_activity_summary_widgets_from_params(self._slope_params)
        if default_mode not in {"ttest", "slope"}:
            default_mode = "ttest"
        if default_mode == "slope":
            self.set_slope_params(self._slope_params)
        self._analysis_mode.setCurrentText(default_mode)
        self._on_analysis_mode_changed(self._analysis_mode.currentText())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_params(self) -> ConditionTestParams:
        """Parse current widget values into a ``ConditionTestParams``.

        Raises
        ------
        ValueError
            When Pydantic validation fails. A ``QMessageBox`` is also shown
            to the user.
        """
        anchor_codes = [
            code.strip()
            for code in self._anchor_codes.text().split(",")
            if code.strip()
        ]
        atlas_name = self._atlas_name.text().strip() or None
        atlas_regions = [
            r.strip()
            for r in self._atlas_regions.text().split(",")
            if r.strip()
        ]
        try:
            params = _merge_model_params(
                self._ttest_params,
                anchor_event_codes=anchor_codes,
                tmin_s=self._tmin.value(),
                tmax_s=self._tmax.value(),
                condition_a=self._condition_a.text().strip(),
                condition_b=self._condition_b.text().strip(),
                min_trials_per_condition=self._min_trials.value(),
                drop_partial_epochs=self._drop_partial.isChecked(),
                equal_var=self._equal_var.isChecked(),
                p_value_correction_method=self._correction.currentText(),
                significance_alpha=self._alpha.value(),
                atlas_name=atlas_name,
                atlas_regions=atlas_regions,
                window_ms=self._window_ms.value(),
                n_bins=self._n_bins.value(),
                activity_zscore=self._activity_scaling.currentText(),
                activity_baseline_tmin_s=self._activity_baseline_tmin.value(),
                activity_baseline_tmax_s=self._activity_baseline_tmax.value(),
                activity_baseline_scope=self._activity_baseline_scope.currentText(),
                activity_baseline_remove_outlier_trial_means=self._activity_baseline_remove_outliers.isChecked(),
                channel_significance_mode=self._sig_mode.currentText(),
                channel_significance_duration_threshold_ms=self._sig_duration_threshold.value(),
            )
            assert isinstance(params, ConditionTestParams)
            self._ttest_params = params
            return params
        except (ValidationError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid parameters", str(exc))
            raise ValueError(str(exc)) from exc

    def get_mode_and_params(self) -> tuple[str, ConditionTestParams | RegressionParams]:
        """Return the selected analysis mode and its validated params object."""
        mode = self._analysis_mode.currentText().strip().lower()
        if mode == "slope":
            return mode, self.get_slope_params()
        return "ttest", self.get_params()

    def get_slope_params(self) -> RegressionParams:
        """Parse current widget values into a ``RegressionParams``."""
        anchor_codes = [
            code.strip()
            for code in self._anchor_codes.text().split(",")
            if code.strip()
        ]
        atlas_name = self._atlas_name.text().strip() or None
        atlas_regions = [
            r.strip()
            for r in self._atlas_regions.text().split(",")
            if r.strip()
        ]
        transform_text = self._predictor_transform.text().strip()
        try:
            predictor_transform_by_condition = (
                json.loads(transform_text) if transform_text else {}
            )
        except json.JSONDecodeError as exc:
            QMessageBox.warning(
                self,
                "Invalid parameters",
                "Predictor transform must be valid JSON shaped like "
                '{"condition":{"scale":1,"offset":0}}.',
            )
            raise ValueError("Invalid predictor transform JSON.") from exc
        try:
            params = _merge_model_params(
                self._slope_params,
                anchor_event_codes=anchor_codes,
                tmin_s=self._tmin.value(),
                tmax_s=self._tmax.value(),
                condition_a=self._condition_a.text().strip(),
                condition_b=self._condition_b.text().strip(),
                min_trials_per_condition=max(3, self._min_trials.value()),
                drop_partial_epochs=self._drop_partial.isChecked(),
                predictor=self._predictor.text().strip(),
                predictor_transform_by_condition=predictor_transform_by_condition,
                predictor_zscore=self._predictor_scaling.currentText(),
                p_value_correction_method=self._correction.currentText(),
                significance_alpha=self._alpha.value(),
                atlas_name=atlas_name,
                atlas_regions=atlas_regions,
                window_ms=self._window_ms.value(),
                n_bins=self._n_bins.value(),
                activity_zscore=self._activity_scaling.currentText(),
                activity_baseline_tmin_s=self._activity_baseline_tmin.value(),
                activity_baseline_tmax_s=self._activity_baseline_tmax.value(),
                activity_baseline_scope=self._activity_baseline_scope.currentText(),
                activity_baseline_remove_outlier_trial_means=self._activity_baseline_remove_outliers.isChecked(),
                trial_activity_summary=self._build_trial_activity_summary_payload(),
            )
            assert isinstance(params, RegressionParams)
            self._slope_params = params
            return params
        except (ValidationError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid parameters", str(exc))
            raise ValueError(str(exc)) from exc

    def set_params(self, params: ConditionTestParams) -> None:
        """Populate all widgets from a ``ConditionTestParams`` instance."""
        self._ttest_params = params
        self._anchor_codes.setText(", ".join(params.anchor_event_codes))
        self._tmin.setValue(params.tmin_s)
        self._tmax.setValue(params.tmax_s)
        self._condition_a.setText(params.condition_a)
        self._condition_b.setText(params.condition_b)
        self._min_trials.setValue(params.min_trials_per_condition)
        self._drop_partial.setChecked(params.drop_partial_epochs)
        self._equal_var.setChecked(params.equal_var)
        idx = self._correction.findText(params.p_value_correction_method)
        if idx >= 0:
            self._correction.setCurrentIndex(idx)
        self._alpha.setValue(params.significance_alpha)
        self._atlas_name.setText(params.atlas_name or "")
        self._atlas_regions.setText(", ".join(params.atlas_regions))
        self._window_ms.setValue(params.window_ms)
        self._n_bins.setValue(params.n_bins)
        self._refresh_activity_scaling_options(is_slope=False)
        idx_activity = self._activity_scaling.findText(params.activity_zscore)
        if idx_activity >= 0:
            self._activity_scaling.setCurrentIndex(idx_activity)
        self._activity_baseline_tmin.setValue(params.activity_baseline_tmin_s)
        self._activity_baseline_tmax.setValue(params.activity_baseline_tmax_s)
        baseline_scope = getattr(params, "activity_baseline_scope", "global")
        idx_baseline_scope = self._activity_baseline_scope.findText(str(baseline_scope))
        if idx_baseline_scope >= 0:
            self._activity_baseline_scope.setCurrentIndex(idx_baseline_scope)
        self._activity_baseline_remove_outliers.setChecked(
            bool(
                getattr(
                    params,
                    "activity_baseline_remove_outlier_trial_means",
                    False,
                )
            )
        )
        idx = self._sig_mode.findText(params.channel_significance_mode)
        if idx >= 0:
            self._sig_mode.setCurrentIndex(idx)
        self._sig_duration_threshold.setValue(
            params.channel_significance_duration_threshold_ms
        )
        self._sig_duration_threshold.setEnabled(
            params.channel_significance_mode == "duration"
        )
        self._analysis_mode.setCurrentText("ttest")

    def set_slope_params(self, params: RegressionParams) -> None:
        """Populate shared + slope-specific widgets from slope params."""
        self._slope_params = params
        self._anchor_codes.setText(", ".join(params.anchor_event_codes))
        self._tmin.setValue(params.tmin_s)
        self._tmax.setValue(params.tmax_s)
        self._condition_a.setText(params.condition_a)
        self._condition_b.setText(params.condition_b)
        self._min_trials.setValue(max(3, params.min_trials_per_condition))
        self._drop_partial.setChecked(params.drop_partial_epochs)
        self._predictor.setText(params.predictor)
        self._predictor_transform.setText(
            _predictor_transform_to_json(params.predictor_transform_by_condition)
        )
        idx_scale = self._predictor_scaling.findText(params.predictor_zscore)
        if idx_scale >= 0:
            self._predictor_scaling.setCurrentIndex(idx_scale)
        idx_corr = self._correction.findText(params.p_value_correction_method)
        if idx_corr >= 0:
            self._correction.setCurrentIndex(idx_corr)
        self._alpha.setValue(params.significance_alpha)
        self._atlas_name.setText(params.atlas_name or "")
        self._atlas_regions.setText(", ".join(params.atlas_regions))
        self._window_ms.setValue(params.window_ms)
        self._n_bins.setValue(params.n_bins)
        self._refresh_activity_scaling_options(is_slope=True)
        idx_activity = self._activity_scaling.findText(params.activity_zscore)
        if idx_activity >= 0:
            self._activity_scaling.setCurrentIndex(idx_activity)
        self._activity_baseline_tmin.setValue(params.activity_baseline_tmin_s)
        self._activity_baseline_tmax.setValue(params.activity_baseline_tmax_s)
        idx_baseline_scope = self._activity_baseline_scope.findText(
            params.activity_baseline_scope
        )
        if idx_baseline_scope >= 0:
            self._activity_baseline_scope.setCurrentIndex(idx_baseline_scope)
        self._activity_baseline_remove_outliers.setChecked(
            params.activity_baseline_remove_outlier_trial_means
        )
        self._set_trial_activity_summary_widgets_from_params(params)
        self._on_activity_scaling_changed(self._activity_scaling.currentText())
        self._refresh_trial_activity_summary_controls()

    @property
    def analysis_mode(self) -> str:
        return self._analysis_mode.currentText().strip().lower() or "ttest"

    def set_status(self, message: str) -> None:
        """Update the status label text."""
        self._status_label.setText(message)

    def set_computing(self, is_computing: bool) -> None:
        """Toggle the Compute button and update the status label."""
        self._compute_btn.setEnabled(not is_computing)
        if is_computing:
            self._status_label.setText("Computing…")

    def set_save_enabled(self, enabled: bool) -> None:
        """Enable or disable the Save results button."""
        self._save_btn.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _on_sig_mode_changed(self, mode: str) -> None:
        self._sig_duration_threshold.setEnabled(mode == "duration")

    def _on_analysis_mode_changed(self, mode: str) -> None:
        is_slope = mode.strip().lower() == "slope"
        self._set_form_row_visible(self._predictor, is_slope)
        self._set_form_row_visible(self._predictor_transform, is_slope)
        self._set_form_row_visible(self._predictor_scaling, is_slope)
        self._set_form_row_visible(self._trial_activity_summary_kind, is_slope)
        self._set_form_row_visible(self._activity_baseline_scope, True)
        self._set_form_row_visible(self._activity_baseline_remove_outliers, True)
        self._set_form_row_visible(self._equal_var, not is_slope)
        self._set_form_row_visible(self._sig_mode, not is_slope)
        self._set_form_row_visible(self._sig_duration_threshold, not is_slope)
        if is_slope and self._correction.currentText() == "permutation":
            self._correction.setCurrentText("fdr_bh")
        self._refresh_activity_scaling_options(is_slope=is_slope)
        self._on_activity_scaling_changed(self._activity_scaling.currentText())
        self._refresh_trial_activity_summary_controls()

    def _on_activity_scaling_changed(self, scaling: str) -> None:
        enabled = scaling.strip().lower() == "baseline"
        self._activity_baseline_tmin.setEnabled(enabled)
        self._activity_baseline_tmax.setEnabled(enabled)
        self._activity_baseline_scope.setEnabled(enabled)
        self._activity_baseline_remove_outliers.setEnabled(enabled)

    def _refresh_activity_scaling_options(self, *, is_slope: bool) -> None:
        current = self._activity_scaling.currentText().strip()
        options = ["none", "baseline"]
        if is_slope:
            options.append("across_trials")
        self._activity_scaling.blockSignals(True)
        self._activity_scaling.clear()
        for option in options:
            self._activity_scaling.addItem(option)
        idx = self._activity_scaling.findText(current if current in options else "none")
        if idx >= 0:
            self._activity_scaling.setCurrentIndex(idx)
        self._activity_scaling.blockSignals(False)

    def _on_trial_activity_summary_kind_changed(self, _: str) -> None:
        self._refresh_trial_activity_summary_controls()

    def _on_trial_activity_summary_source_changed(self, _: str) -> None:
        self._refresh_trial_activity_summary_controls()

    def _on_window_ms_changed(self, value: float) -> None:
        if value > 0.0 and self._n_bins.value() > 0:
            self._n_bins.blockSignals(True)
            self._n_bins.setValue(0)
            self._n_bins.blockSignals(False)

    def _on_n_bins_changed(self, value: int) -> None:
        if value > 0 and self._window_ms.value() > 0.0:
            self._window_ms.blockSignals(True)
            self._window_ms.setValue(0.0)
            self._window_ms.blockSignals(False)

    def _set_form_row_visible(self, field_widget: QWidget, visible: bool) -> None:
        label_widget = self._form.labelForField(field_widget)
        if label_widget is not None:
            label_widget.setVisible(visible)
        field_widget.setVisible(visible)

    def _build_trial_activity_summary_payload(self) -> dict[str, object]:
        kind = self._trial_activity_summary_kind.currentText().strip()
        payload: dict[str, object] = {
            "kind": kind,
            "missing_response_policy": "clamp_to_epoch",
        }
        if kind != "anchor_to_response_mean":
            payload["response"] = None
            return payload

        source = self._trial_activity_summary_source.currentText().strip()
        if source == "table_column":
            payload["response"] = {
                "source": source,
                "column": self._trial_activity_summary_response_column.text().strip(),
                "units": self._trial_activity_summary_response_units.currentText().strip(),
            }
        else:
            payload["response"] = {
                "source": source,
                "event_code": self._trial_activity_summary_response_event_code.text().strip(),
                "occurrence": "first_after_anchor",
            }
        return payload

    def _set_trial_activity_summary_widgets_from_params(
        self,
        params: RegressionParams,
    ) -> None:
        summary = params.trial_activity_summary
        idx_kind = self._trial_activity_summary_kind.findText(summary.kind)
        if idx_kind >= 0:
            self._trial_activity_summary_kind.setCurrentIndex(idx_kind)

        response = summary.response
        if response is not None:
            idx_source = self._trial_activity_summary_source.findText(response.source)
            if idx_source >= 0:
                self._trial_activity_summary_source.setCurrentIndex(idx_source)
            if response.source == "table_column":
                self._trial_activity_summary_response_column.setText(response.column)
                idx_units = self._trial_activity_summary_response_units.findText(response.units)
                if idx_units >= 0:
                    self._trial_activity_summary_response_units.setCurrentIndex(idx_units)
                self._trial_activity_summary_response_event_code.clear()
            else:
                self._trial_activity_summary_response_event_code.setText(response.event_code)
        else:
            self._trial_activity_summary_source.setCurrentText("table_column")
            self._trial_activity_summary_response_column.clear()
            self._trial_activity_summary_response_units.setCurrentText("s")
            self._trial_activity_summary_response_event_code.clear()

    def _refresh_trial_activity_summary_controls(self) -> None:
        is_slope = self.analysis_mode == "slope"
        summary_kind = self._trial_activity_summary_kind.currentText().strip()
        show_anchor_controls = is_slope and summary_kind == "anchor_to_response_mean"
        source = self._trial_activity_summary_source.currentText().strip()
        show_table_controls = show_anchor_controls and source == "table_column"
        show_annotation_controls = (
            show_anchor_controls and source == "annotation_event_code"
        )

        self._set_form_row_visible(self._trial_activity_summary_source, show_anchor_controls)
        self._set_form_row_visible(
            self._trial_activity_summary_response_column,
            show_table_controls,
        )
        self._set_form_row_visible(
            self._trial_activity_summary_response_units,
            show_table_controls,
        )
        self._set_form_row_visible(
            self._trial_activity_summary_response_event_code,
            show_annotation_controls,
        )



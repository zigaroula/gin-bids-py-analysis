"""Right panel: TrialStatsParams form + Compute button."""

from __future__ import annotations

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

from gin_bids_py_analysis.processing.trial_stats import TrialStatsParams


class ParamsPanel(QWidget):
    """Right panel with all ``TrialStatsParams`` fields and a Compute button.

    Signals
    -------
    compute_requested : emitted when the user clicks *Compute*.
    save_requested    : emitted when the user clicks *Save results*.
    """

    compute_requested = Signal()
    save_requested = Signal()

    def __init__(
        self,
        params: TrialStatsParams,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumWidth(280)
        self.setMaximumWidth(400)

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

        # Mutual exclusion: setting one to nonzero zeros the other
        self._window_ms.valueChanged.connect(self._on_window_ms_changed)
        self._n_bins.valueChanged.connect(self._on_n_bins_changed)

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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_params(self) -> TrialStatsParams:
        """Parse current widget values into a ``TrialStatsParams``.

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
            return TrialStatsParams(
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
            )
        except (ValidationError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid parameters", str(exc))
            raise ValueError(str(exc)) from exc

    def set_params(self, params: TrialStatsParams) -> None:
        """Populate all widgets from a ``TrialStatsParams`` instance."""
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

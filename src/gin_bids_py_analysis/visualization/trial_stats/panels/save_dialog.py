"""Dialog for configuring ConditionTestWriterParams before saving results."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from gin_bids_py_analysis.processing.trial_stats import ConditionTestWriterParams


class SaveTrialStatsDialog(QDialog):
    """Dialog that lets the user configure writer params for subject-level results.

    The ``bids_root`` is pre-filled and read-only (it is always derived from the
    input dataset).  The user may choose the output format and whether to include
    per-trial epoch arrays.

    Parameters
    ----------
    bids_root:
        Root of the BIDS dataset.  Shown read-only so the user knows where files
        will be written.
    parent:
        Optional parent widget.
    """

    def __init__(
        self,
        bids_root: Path,
        parent: QWidget | None = None,
        *,
        default_pipeline_label: str = "condition_test",
        default_output_description: str = "conditiontest",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save results")
        self.setMinimumWidth(380)

        self._bids_root = bids_root

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)

        # bids_root — read-only
        self._root_edit = QLineEdit(str(bids_root))
        self._root_edit.setReadOnly(True)
        self._root_edit.setStyleSheet("color: gray;")
        form.addRow("BIDS root", self._root_edit)

        # pipeline_label
        self._pipeline_label = QLineEdit(default_pipeline_label)
        form.addRow("Pipeline label", self._pipeline_label)

        # output_modality
        self._output_modality = QLineEdit("ieeg")
        form.addRow("Output modality", self._output_modality)

        # output_suffix
        self._output_suffix = QLineEdit("stats")
        form.addRow("Output suffix", self._output_suffix)

        # output_description
        self._output_description = QLineEdit(default_output_description)
        form.addRow("Output description", self._output_description)

        # output_format
        self._format_combo = QComboBox()
        for fmt in ("hdf5", "matlab"):
            self._format_combo.addItem(fmt)
        form.addRow("Output format", self._format_combo)

        # include_epochs
        self._include_epochs = QCheckBox()
        self._include_epochs.setToolTip(
            "When checked, per-trial epoch arrays are written to the output file. "
            "This can significantly increase file size."
        )
        form.addRow("Include epochs", self._include_epochs)

        layout.addLayout(form)

        # Info label
        info = QLabel(
            "Results for all computed subjects will be saved to:\n"
            f"  {bids_root / 'derivatives' / '<pipeline_label>'}"
        )
        info.setStyleSheet("color: gray; font-size: 11px;")
        info.setWordWrap(True)
        layout.addWidget(info)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_writer_params(self) -> ConditionTestWriterParams:
        """Build a :class:`ConditionTestWriterParams` from the current dialog values."""
        return ConditionTestWriterParams(**self.get_common_writer_kwargs())

    def get_common_writer_kwargs(self) -> dict[str, object]:
        """Return dialog values as kwargs accepted by trial-stats writers."""
        return {
            "bids_root": self._bids_root,
            "pipeline_label": self._pipeline_label.text().strip(),
            "output_modality": self._output_modality.text().strip(),
            "output_suffix": self._output_suffix.text().strip(),
            "output_description": self._output_description.text().strip(),
            "output_format": self._format_combo.currentText(),
            "include_epochs": self._include_epochs.isChecked(),
        }



"""Dialog for configuring TrialStatsGroupWriterParams before saving group results."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from gin_bids_py_analysis.processing.trial_stats_group import TrialStatsGroupWriterParams


class SaveTrialStatsGroupDialog(QDialog):
    """Dialog that lets the user configure writer params for group-level results.

    The ``bids_root`` is pre-filled and read-only (it is always derived from the
    input dataset).  The user may choose the output format.

    Parameters
    ----------
    bids_root:
        Root of the BIDS dataset.  Shown read-only so the user knows where files
        will be written.
    parent:
        Optional parent widget.
    """

    def __init__(self, bids_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Save group results")
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
        self._pipeline_label = QLineEdit("trial_stats_group")
        form.addRow("Pipeline label", self._pipeline_label)

        # output_modality
        self._output_modality = QLineEdit("ieeg")
        form.addRow("Output modality", self._output_modality)

        # output_suffix
        self._output_suffix = QLineEdit("stats")
        form.addRow("Output suffix", self._output_suffix)

        # output_description
        self._output_description = QLineEdit("trialstatsgroup")
        form.addRow("Output description", self._output_description)

        # output_format
        self._format_combo = QComboBox()
        for fmt in ("hdf5", "matlab"):
            self._format_combo.addItem(fmt)
        form.addRow("Output format", self._format_combo)

        layout.addLayout(form)

        # Info label
        info = QLabel(
            "The group result will be saved to:\n"
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

    def get_writer_params(self) -> TrialStatsGroupWriterParams:
        """Build a :class:`TrialStatsGroupWriterParams` from the current dialog values."""
        return TrialStatsGroupWriterParams(
            bids_root=self._bids_root,
            pipeline_label=self._pipeline_label.text().strip(),
            output_modality=self._output_modality.text().strip(),
            output_suffix=self._output_suffix.text().strip(),
            output_description=self._output_description.text().strip(),
            output_format=self._format_combo.currentText(),
        )

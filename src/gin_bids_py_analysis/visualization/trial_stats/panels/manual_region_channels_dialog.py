"""Dialog for interactively editing the manual_region_channels mapping."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Qt


class ManualRegionChannelsDialog(QDialog):
    """Dialog for editing ``manual_region_channels``.

    Layout
    ------
    Left panel : ``QListWidget`` of region names + **+** / **−** buttons.
    Right panel : One row per subject — subject label and a ``QLineEdit``
                  accepting comma-separated channel names.  All *subject_ids*
                  are always shown; subjects present in *mapping* but absent
                  from *subject_ids* (e.g. from a previous script run) are
                  appended below.

    Parameters
    ----------
    mapping:
        Current ``{region: {subject_id: [channel, …]}}`` mapping.  A deep
        copy is made on entry so *Cancel* discards all edits.
    subject_ids:
        Ordered list of subject IDs to show as rows in the right panel.
    """

    def __init__(
        self,
        mapping: dict[str, dict[str, list[str]]],
        subject_ids: list[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Edit manual region channels")
        self.resize(720, 500)
        self.setMinimumSize(520, 360)

        self._subject_ids: list[str] = list(subject_ids)
        # Deep copy so Cancel leaves the original untouched
        self._mapping: dict[str, dict[str, list[str]]] = {
            region: dict(subj_map) for region, subj_map in mapping.items()
        }
        self._current_region: str | None = None
        self._subject_edits: dict[str, QLineEdit] = {}

        # ------------------------------------------------------------------
        # Outer layout: splitter on top, button box at bottom
        # ------------------------------------------------------------------
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, stretch=1)

        # ------------------------------------------------------------------
        # Left panel — region list
        # ------------------------------------------------------------------
        left = QWidget()
        left.setMinimumWidth(140)
        left.setMaximumWidth(200)
        left_vbox = QVBoxLayout(left)
        left_vbox.setContentsMargins(0, 0, 4, 0)
        left_vbox.setSpacing(4)

        left_vbox.addWidget(QLabel("<b>Regions</b>"))
        self._region_list = QListWidget()
        left_vbox.addWidget(self._region_list, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self._add_btn = QPushButton("+")
        self._add_btn.setToolTip("Add new region")
        self._add_btn.setFixedSize(28, 28)
        self._del_btn = QPushButton("−")
        self._del_btn.setToolTip("Delete selected region")
        self._del_btn.setFixedSize(28, 28)
        btn_row.addWidget(self._add_btn)
        btn_row.addWidget(self._del_btn)
        btn_row.addStretch()
        left_vbox.addLayout(btn_row)

        splitter.addWidget(left)

        # ------------------------------------------------------------------
        # Right panel — per-subject channel inputs
        # ------------------------------------------------------------------
        right = QWidget()
        right_vbox = QVBoxLayout(right)
        right_vbox.setContentsMargins(4, 0, 0, 0)
        right_vbox.setSpacing(6)

        self._region_title = QLabel("← Select a region")
        self._region_title.setStyleSheet("font-weight: bold; font-size: 13px;")
        right_vbox.addWidget(self._region_title)

        hint = QLabel("Enter channel names separated by commas (e.g. Y02, Y06).")
        hint.setStyleSheet("color: gray; font-size: 11px;")
        right_vbox.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._subject_form_widget = QWidget()
        self._subject_form_layout = QFormLayout(self._subject_form_widget)
        self._subject_form_layout.setContentsMargins(0, 4, 4, 4)
        self._subject_form_layout.setSpacing(6)
        scroll.setWidget(self._subject_form_widget)
        right_vbox.addWidget(scroll, stretch=1)

        splitter.addWidget(right)
        splitter.setSizes([160, 540])

        # ------------------------------------------------------------------
        # Dialog buttons
        # ------------------------------------------------------------------
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self.reject)
        outer.addWidget(button_box)

        # ------------------------------------------------------------------
        # Wire signals
        # ------------------------------------------------------------------
        self._add_btn.clicked.connect(self._on_add_region)
        self._del_btn.clicked.connect(self._on_delete_region)
        self._region_list.currentRowChanged.connect(self._on_region_changed)

        # Populate region list
        self._populate_region_list()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def result_mapping(self) -> dict[str, dict[str, list[str]]]:
        """Return the edited mapping (only valid after ``exec()`` returns ``Accepted``)."""
        return dict(self._mapping)

    # ------------------------------------------------------------------
    # Region list management
    # ------------------------------------------------------------------

    def _populate_region_list(self) -> None:
        self._region_list.blockSignals(True)
        self._region_list.clear()
        for region in self._mapping:
            self._region_list.addItem(region)
        self._region_list.blockSignals(False)
        if self._region_list.count() > 0:
            self._region_list.setCurrentRow(0)

    def _on_add_region(self) -> None:
        name, ok = QInputDialog.getText(self, "New region", "Region name:")
        if not ok:
            return
        name = name.strip()
        if not name:
            return
        if name in self._mapping:
            QMessageBox.warning(self, "Duplicate region", f'"{name}" already exists.')
            return
        self._save_current_region()
        self._mapping[name] = {}
        self._region_list.addItem(name)
        self._region_list.setCurrentRow(self._region_list.count() - 1)

    def _on_delete_region(self) -> None:
        row = self._region_list.currentRow()
        if row < 0:
            return
        region = self._region_list.item(row).text()
        reply = QMessageBox.question(
            self,
            "Delete region",
            f'Delete region "{region}" and all its channel assignments?',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        # Prevent _save_current_region from re-inserting the deleted region
        self._current_region = None
        self._mapping.pop(region, None)
        self._region_list.takeItem(row)
        # takeItem changes selection → _on_region_changed fires automatically

    # ------------------------------------------------------------------
    # Right-panel population
    # ------------------------------------------------------------------

    def _on_region_changed(self, row: int) -> None:
        if row < 0:
            self._clear_subject_form()
            self._region_title.setText("← Select a region")
            self._current_region = None
            return
        region = self._region_list.item(row).text()
        if region == self._current_region:
            return
        self._save_current_region()
        self._load_region(region)

    def _load_region(self, region: str) -> None:
        self._current_region = region
        self._region_title.setText(f"Region: {region}")
        self._clear_subject_form()

        region_data = self._mapping.get(region, {})

        # Known subjects first, then any extra subjects from the existing mapping
        seen: set[str] = set(self._subject_ids)
        all_subjects = list(self._subject_ids)
        for subj in region_data:
            if subj not in seen:
                all_subjects.append(subj)
                seen.add(subj)

        self._subject_edits = {}
        for subj in all_subjects:
            edit = QLineEdit()
            edit.setPlaceholderText("CH1, CH2, …")
            channels = region_data.get(subj, [])
            if channels:
                edit.setText(", ".join(channels))
            self._subject_form_layout.addRow(QLabel(subj), edit)
            self._subject_edits[subj] = edit

    def _clear_subject_form(self) -> None:
        while self._subject_form_layout.rowCount() > 0:
            self._subject_form_layout.removeRow(0)
        self._subject_edits = {}

    def _save_current_region(self) -> None:
        if self._current_region is None:
            return
        subj_dict: dict[str, list[str]] = {}
        for subj_id, edit in self._subject_edits.items():
            text = edit.text().strip()
            if text:
                channels = [c.strip() for c in text.split(",") if c.strip()]
                if channels:
                    subj_dict[subj_id] = channels
        self._mapping[self._current_region] = subj_dict

    # ------------------------------------------------------------------
    # Accept / OK
    # ------------------------------------------------------------------

    def _on_accept(self) -> None:
        self._save_current_region()
        # Drop regions that ended up completely empty
        self._mapping = {
            region: subj_map
            for region, subj_map in self._mapping.items()
            if subj_map
        }
        self.accept()

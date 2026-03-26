"""Left panel: subject dropdown + channel / region list."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
    QWidget,
)


class SubjectChannelPanel(QWidget):
    """Left panel with a subject selector and a channel / region list.

    Signals
    -------
    subject_changed : str  — emitted when the subject dropdown value changes.
    channel_changed : int  — emitted with the new row index when the channel
                             selection changes.
    """

    subject_changed = Signal(str)
    channel_changed = Signal(int)

    def __init__(
        self,
        subject_ids: list[str],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumWidth(180)
        self.setMaximumWidth(260)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        layout.addWidget(QLabel("Subject"))
        self._subject_combo = QComboBox()
        for subj in sorted(subject_ids):
            self._subject_combo.addItem(subj)
        layout.addWidget(self._subject_combo)

        layout.addSpacing(8)
        layout.addWidget(QLabel("Channels / Regions"))
        self._channel_list = QListWidget()
        layout.addWidget(self._channel_list, stretch=1)

        self._subject_combo.currentTextChanged.connect(self.subject_changed)
        self._channel_list.currentRowChanged.connect(self.channel_changed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_channels(self, channel_names: list[str], restore_name: str | None = None) -> None:
        """Repopulate the channel list.

        If *restore_name* is given and exists in *channel_names*, that channel
        is re-selected; otherwise the first entry is selected.
        """
        self._channel_list.blockSignals(True)
        self._channel_list.clear()
        for name in channel_names:
            self._channel_list.addItem(QListWidgetItem(name))
        self._channel_list.blockSignals(False)
        if self._channel_list.count() > 0:
            restore_row = 0
            if restore_name is not None:
                matches = self._channel_list.findItems(restore_name, Qt.MatchFlag.MatchExactly)
                if matches:
                    restore_row = self._channel_list.row(matches[0])
            self._channel_list.setCurrentRow(restore_row)

    def set_interactive(self, enabled: bool) -> None:
        """Enable or disable subject and channel selection."""
        self._subject_combo.setEnabled(enabled)
        self._channel_list.setEnabled(enabled)

    @property
    def current_subject(self) -> str:
        return self._subject_combo.currentText()

    @property
    def current_channel_index(self) -> int:
        return max(self._channel_list.currentRow(), 0)

    @property
    def current_channel_name(self) -> str | None:
        """Name of the currently selected channel, or ``None`` if the list is empty."""
        item = self._channel_list.currentItem()
        return item.text() if item is not None else None

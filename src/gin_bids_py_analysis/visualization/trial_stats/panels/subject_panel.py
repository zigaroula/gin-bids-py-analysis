"""Left panel: subject dropdown + channel / region list."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QStyledItemDelegate,
    QVBoxLayout,
    QWidget,
)


class _SignificanceDelegate(QStyledItemDelegate):
    """Paints a small filled orange square at the right edge of flagged rows."""

    _COLOR = QColor("darkorange")
    _SIZE = 10
    _MARGIN = 5

    def __init__(self, significant_rows: frozenset[int], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._significant_rows = significant_rows

    def paint(self, painter, option, index) -> None:  # type: ignore[override]
        super().paint(painter, option, index)
        if index.row() in self._significant_rows:
            painter.save()
            rect = option.rect
            side = self._SIZE
            x = rect.right() - side - self._MARGIN
            y = rect.top() + (rect.height() - side) // 2
            painter.fillRect(x, y, side, side, self._COLOR)
            painter.restore()


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

    def set_channels(
        self,
        channel_names: list[str],
        restore_name: str | None = None,
        channel_significant_mask: np.ndarray | None = None,
    ) -> None:
        """Repopulate the channel list.

        If *restore_name* is given and exists in *channel_names*, that channel
        is re-selected; otherwise the first entry is selected.

        If *channel_significant_mask* is provided (bool array, shape
        ``(n_channels,)``), channels for which the mask is ``True`` are
        marked with a small filled orange square on the right edge of the row.
        """
        if channel_significant_mask is not None:
            significant_rows = frozenset(
                int(i)
                for i in range(len(channel_names))
                if bool(channel_significant_mask[i])
            )
        else:
            significant_rows = frozenset()

        self._channel_list.setItemDelegate(
            _SignificanceDelegate(significant_rows, parent=self._channel_list)
        )
        self._channel_list.blockSignals(True)
        self._channel_list.clear()
        for name in channel_names:
            self._channel_list.addItem(QListWidgetItem(name))
        self._channel_list.blockSignals(False)
        if significant_rows:
            self._channel_list.setToolTip("Channels marked with an orange square passed the channel-level significance test")
        else:
            self._channel_list.setToolTip("")
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

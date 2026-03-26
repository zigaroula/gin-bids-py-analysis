"""Background computation and preloading workers for trial statistics."""

from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from gin_bids_py_analysis.bids import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_stats import TrialStatsProcessing


class PreloadWorker(QThread):
    """Pre-load all ``BIDSFile`` objects across every subject group in a background thread.

    This worker calls :meth:`~gin_bids_py_analysis.bids.BIDSFile.preload` on
    each file exactly once.  After it completes, every file is permanently
    attached in memory so that repeated computations (e.g. parameter sweeps on
    the same subject) do not re-read data from disk.

    Signals
    -------
    progress : str  — human-readable progress message after each file load.
    finished :       — emitted when all files have been loaded successfully.
    error    : str  — emitted with the exception message if any file fails to load.
    """

    progress = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(
        self,
        subject_groups: dict[str, BIDSFileGroup],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._subject_groups = subject_groups

    def run(self) -> None:
        all_files = [
            file
            for group in self._subject_groups.values()
            for file in group.all_files
        ]
        total = len(all_files)
        try:
            for i, file in enumerate(all_files, start=1):
                if not file.is_loaded:
                    self.progress.emit(
                        f"Pre-loading {file.path.name}  ({i}/{total})…"
                    )
                    file.preload()
            self.finished.emit()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class ComputeWorker(QThread):
    """Run TrialStatsProcessing in a background thread.

    Parameters
    ----------
    processor:
        A freshly constructed ``TrialStatsProcessing`` instance.
    group:
        The ``BIDSFileGroup`` for the subject to process.

    Signals
    -------
    result_ready : emitted with the ``TrialStatsProcessingResult`` on success.
    error        : emitted with the exception message string on failure.
    """

    result_ready = Signal(object)  # TrialStatsProcessingResult
    error = Signal(str)

    def __init__(
        self,
        processor: TrialStatsProcessing,
        group: BIDSFileGroup,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._processor = processor
        self._group = group

    def run(self) -> None:
        try:
            result = self._processor.process_group(self._group)
            self.result_ready.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))

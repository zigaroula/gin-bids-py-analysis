"""Background computation and preloading workers for trial statistics."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from PySide6.QtCore import QThread, Signal

from gin_bids_py_analysis.bids import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_stats import (
    TrialStatsParams,
    TrialStatsProcessing,
)

if TYPE_CHECKING:
    from gin_bids_py_analysis.processing.trial_stats.result import (
        TrialStatsProcessingResult,
    )
    from gin_bids_py_analysis.processing.trial_stats_group.params import (
        TrialStatsGroupParams,
    )


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


class ComputeAllWorker(QThread):
    """Run TrialStatsProcessing for every subject group sequentially in a background thread.

    Parameters
    ----------
    processor_factory:
        Callable that accepts a ``TrialStatsParams`` and returns a fresh
        ``TrialStatsProcessing`` instance.  Called once per subject.
    subject_groups:
        Mapping of subject id to ``BIDSFileGroup``.
    params:
        Parameters to pass to the factory for each subject.

    Signals
    -------
    subject_done : (str, object) — emitted after each subject with (subject_id, result).
    all_done     : (object)      — emitted with the full ``dict[str, result]`` when every
                                   subject has been processed.
    progress     : str           — human-readable progress message.
    error        : str           — emitted with the exception message on failure.
    """

    subject_done = Signal(str, object)   # subject_id, TrialStatsProcessingResult
    all_done = Signal(object)            # dict[str, TrialStatsProcessingResult]
    progress = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        processor_factory: Callable[[TrialStatsParams], TrialStatsProcessing],
        subject_groups: dict[str, BIDSFileGroup],
        params: TrialStatsParams,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._factory = processor_factory
        self._subject_groups = subject_groups
        self._params = params

    def run(self) -> None:
        results: dict[str, TrialStatsProcessingResult] = {}
        total = len(self._subject_groups)
        try:
            for i, (subject_id, group) in enumerate(self._subject_groups.items(), start=1):
                self.progress.emit(f"Computing subject {subject_id}  ({i}/{total})…")
                processor = self._factory(self._params)
                result = processor.process_group(group)
                results[subject_id] = result
                self.subject_done.emit(subject_id, result)
            self.all_done.emit(results)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class GroupComputeWorker(QThread):
    """Run TrialStatsGroupProcessing from in-memory subject results in a background thread.

    Parameters
    ----------
    all_results:
        Mapping of subject id to ``TrialStatsProcessingResult``.
    group_params:
        Parameters for the group-level analysis.

    Signals
    -------
    result_ready : emitted with the ``TrialStatsGroupProcessingResult`` on success.
    error        : emitted with the exception message string on failure.
    """

    result_ready = Signal(object)   # TrialStatsGroupProcessingResult
    error = Signal(str)

    def __init__(
        self,
        all_results: dict[str, "TrialStatsProcessingResult"],
        group_params: "TrialStatsGroupParams",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._all_results = all_results
        self._group_params = group_params

    def run(self) -> None:
        try:
            from gin_bids_py_analysis.processing.trial_stats_group import (
                TrialStatsGroupProcessing,
            )

            from ._bridge import group_file_group_context

            processor = TrialStatsGroupProcessing(self._group_params)
            with group_file_group_context(
                self._all_results,
                source_metric=self._group_params.source_metric,
            ) as group:
                result = processor.process_group(group)
            self.result_ready.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))

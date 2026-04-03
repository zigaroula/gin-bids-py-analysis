"""Background computation and preloading workers for trial statistics."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from PySide6.QtCore import QThread, Signal

from gin_bids_py_analysis.bids import BIDSFileGroup
from gin_bids_py_analysis.processing.trial_slope_stats import (
    TrialSlopeStatsParams,
    TrialSlopeStatsProcessing,
)
from gin_bids_py_analysis.processing.trial_stats import (
    TrialStatsParams,
    TrialStatsProcessing,
)

if TYPE_CHECKING:
    from gin_bids_py_analysis.processing.trial_stats.result import (
        TrialStatsProcessingResult,
    )
    from gin_bids_py_analysis.processing.trial_stats.writer import (
        TrialStatsProcessingWriter,
    )
    from gin_bids_py_analysis.processing.trial_stats_group.params import (
        TrialStatsGroupParams,
    )
    from gin_bids_py_analysis.processing.trial_stats_group.result import (
        TrialStatsGroupProcessingResult,
    )
    from gin_bids_py_analysis.processing.trial_stats_group.writer import (
        TrialStatsGroupProcessingWriter,
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
        processor: TrialStatsProcessing | TrialSlopeStatsProcessing,
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
        processor_factory: Callable[
            [TrialStatsParams | TrialSlopeStatsParams],
            TrialStatsProcessing | TrialSlopeStatsProcessing,
        ],
        subject_groups: dict[str, BIDSFileGroup],
        params: TrialStatsParams | TrialSlopeStatsParams,
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


class WriteAllWorker(QThread):
    """Write all per-subject ``TrialStatsProcessingResult`` objects in a background thread.

    Parameters
    ----------
    results:
        Mapping of subject id to ``TrialStatsProcessingResult``.
    writer:
        A fully configured ``TrialStatsProcessingWriter`` instance.

    Signals
    -------
    progress : str  — human-readable progress message after each subject is written.
    finished : int  — emitted with the total number of subjects written on success.
    error    : str  — emitted with the exception message on failure.
    """

    progress = Signal(str)
    finished = Signal(int)
    error = Signal(str)

    def __init__(
        self,
        results: "dict[str, TrialStatsProcessingResult]",
        writer: "TrialStatsProcessingWriter",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._results = results
        self._writer = writer

    def run(self) -> None:
        total = len(self._results)
        try:
            for i, (subject_id, result) in enumerate(self._results.items(), start=1):
                self.progress.emit(f"Saving subject {subject_id}  ({i}/{total})…")
                self._writer.write(result)
            self.finished.emit(total)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class WriteGroupWorker(QThread):
    """Write a single ``TrialStatsGroupProcessingResult`` in a background thread.

    Parameters
    ----------
    result:
        The group-level result to write.
    writer:
        A fully configured ``TrialStatsGroupProcessingWriter`` instance.

    Signals
    -------
    finished : emitted on success.
    error    : str  — emitted with the exception message on failure.
    """

    finished = Signal()
    error = Signal(str)

    def __init__(
        self,
        result: "TrialStatsGroupProcessingResult",
        writer: "TrialStatsGroupProcessingWriter",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._result = result
        self._writer = writer

    def run(self) -> None:
        try:
            self._writer.write(self._result)
            self.finished.emit()
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class LoadSubjectResultsWorker(QThread):
    """Load pre-computed subject results from files in a background thread.

    This worker reads ``.h5``/``.hdf5`` or ``.mat`` trial-stats files written
    by ``TrialStatsProcessingWriter`` and reconstructs ``TrialStatsProcessingResult``
    objects without re-running any processing.

    Parameters
    ----------
    subject_files:
        Mapping of ``subject_id → Path`` to the stats file for that subject.

    Signals
    -------
    subject_done : (str, object) — emitted after each subject with (subject_id, result).
    all_done     : (object)      — emitted with the full ``dict[str, result]`` when every
                                   subject has been loaded.
    progress     : str           — human-readable progress message.
    error        : str           — emitted with the exception message on failure.
    """

    subject_done = Signal(str, object)  # subject_id, TrialStatsProcessingResult
    all_done = Signal(object)           # dict[str, TrialStatsProcessingResult]
    progress = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        subject_files: dict[str, Path],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._subject_files = subject_files

    def run(self) -> None:
        from gin_bids_py_analysis.processing.trial_slope_stats.result_loader import (
            load_trial_slope_stats_result,
        )
        from gin_bids_py_analysis.processing.trial_stats.result_loader import (
            load_trial_stats_result,
        )

        results: dict[str, "TrialStatsProcessingResult"] = {}
        total = len(self._subject_files)
        try:
            for i, (subject_id, path) in enumerate(self._subject_files.items(), start=1):
                self.progress.emit(
                    f"Loading subject {subject_id}  ({i}/{total})…"
                )
                result = _load_subject_result_auto(
                    path,
                    load_trial_stats_result=load_trial_stats_result,
                    load_trial_slope_stats_result=load_trial_slope_stats_result,
                )
                results[subject_id] = result
                self.subject_done.emit(subject_id, result)
            self.all_done.emit(results)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class LoadGroupResultWorker(QThread):
    """Load a pre-computed group result from a file in a background thread.

    Parameters
    ----------
    group_file:
        Path to the ``.h5``/``.hdf5`` or ``.mat`` group stats file written by
        ``TrialStatsGroupProcessingWriter``.

    Signals
    -------
    result_ready : emitted with the ``TrialStatsGroupProcessingResult`` on success.
    error        : str — emitted with the exception message on failure.
    """

    result_ready = Signal(object)  # TrialStatsGroupProcessingResult
    error = Signal(str)

    def __init__(
        self,
        group_file: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._group_file = group_file

    def run(self) -> None:
        try:
            from gin_bids_py_analysis.processing.trial_stats_group.result_loader import (
                load_trial_stats_group_result,
            )

            result = load_trial_stats_group_result(self._group_file)
            self.result_ready.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


def _load_subject_result_auto(
    path: Path,
    *,
    load_trial_stats_result,
    load_trial_slope_stats_result,
):
    suffix = path.suffix.lower()
    if suffix in {".h5", ".hdf5"}:
        try:
            import h5py

            with h5py.File(path, "r") as fh:
                analysis_type = ""
                if "meta" in fh and "analysis_type" in fh["meta"]:
                    try:
                        analysis_type = str(fh["meta"]["analysis_type"].asstr()[()]).strip()
                    except Exception:
                        analysis_type = ""
            if analysis_type == "slope_regression":
                return load_trial_slope_stats_result(path)
            return load_trial_stats_result(path)
        except Exception:
            # Fallback: try slope loader first, then classic trial-stats.
            try:
                return load_trial_slope_stats_result(path)
            except Exception:
                return load_trial_stats_result(path)

    # MATLAB and others: optimistic slope-first fallback strategy.
    try:
        return load_trial_slope_stats_result(path)
    except Exception:
        return load_trial_stats_result(path)

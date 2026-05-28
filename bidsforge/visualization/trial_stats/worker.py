"""Background computation and preloading workers for trial statistics."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Callable

from PySide6.QtCore import QThread, Signal

from bidsforge.bids import BIDSFileGroup
from bidsforge.processing.trial_stats import (
    ConditionTestParams,
    ConditionTestProcessing,
    RegressionParams,
    RegressionProcessing,
)

if TYPE_CHECKING:
    from bidsforge.processing.trial_stats import (
        RegressionProcessingResult,
    )
    from bidsforge.processing.trial_stats_group import (
        RegressionGroupParams,
    )
    from bidsforge.processing.trial_stats_group import (
        RegressionGroupProcessingResult,
    )
    from bidsforge.processing.trial_stats_group import (
        RegressionGroupProcessingWriter,
    )
    from bidsforge.processing.trial_stats import (
        ConditionTestProcessingResult,
    )
    from bidsforge.processing.trial_stats import (
        ConditionTestProcessingWriter,
    )
    from bidsforge.processing.trial_stats_group import (
        ConditionTestGroupParams,
    )
    from bidsforge.processing.trial_stats_group import (
        ConditionTestGroupProcessingResult,
    )
    from bidsforge.processing.trial_stats_group import (
        ConditionTestGroupProcessingWriter,
    )


class PreloadWorker(QThread):
    """Pre-load all ``BIDSFile`` objects across every subject group in a background thread.

    This worker calls :meth:`~bidsforge.bids.BIDSFile.preload` on
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
    """Run ConditionTestProcessing in a background thread.

    Parameters
    ----------
    processor:
        A freshly constructed ``ConditionTestProcessing`` instance.
    group:
        The ``BIDSFileGroup`` for the subject to process.

    Signals
    -------
    result_ready : emitted with the ``ConditionTestProcessingResult`` on success.
    error        : emitted with the exception message string on failure.
    """

    result_ready = Signal(object)  # ConditionTestProcessingResult
    error = Signal(str)

    def __init__(
        self,
        processor: ConditionTestProcessing | RegressionProcessing,
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
    """Run ConditionTestProcessing for every subject group sequentially in a background thread.

    Parameters
    ----------
    processor_factory:
        Callable that accepts a ``ConditionTestParams`` and returns a fresh
        ``ConditionTestProcessing`` instance.  Called once per subject.
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

    subject_done = Signal(str, object)   # subject_id, ConditionTestProcessingResult
    all_done = Signal(object)            # dict[str, ConditionTestProcessingResult]
    progress = Signal(str)
    error = Signal(str)

    def __init__(
        self,
        processor_factory: Callable[
            [ConditionTestParams | RegressionParams],
            ConditionTestProcessing | RegressionProcessing,
        ],
        subject_groups: dict[str, BIDSFileGroup],
        params: ConditionTestParams | RegressionParams,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._factory = processor_factory
        self._subject_groups = subject_groups
        self._params = params

    def run(self) -> None:
        results: dict[str, ConditionTestProcessingResult] = {}
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
    """Run ConditionTestGroupProcessing from in-memory subject results in a background thread.

    Parameters
    ----------
    all_results:
        Mapping of subject id to ``ConditionTestProcessingResult``.
    group_params:
        Parameters for the group-level analysis.

    Signals
    -------
    result_ready : emitted with the ``ConditionTestGroupProcessingResult`` on success.
    error        : emitted with the exception message string on failure.
    """

    result_ready = Signal(object)   # ConditionTestGroupProcessingResult
    error = Signal(str)

    def __init__(
        self,
        all_results: dict[str, "ConditionTestProcessingResult | RegressionProcessingResult"],
        group_params: "ConditionTestGroupParams | RegressionGroupParams",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._all_results = all_results
        self._group_params = group_params

    def run(self) -> None:
        try:
            if _is_slope_group_params(self._group_params):
                from bidsforge.processing.trial_stats_group import (
                    RegressionGroupProcessing,
                )

                from ._bridge_slope import group_file_group_context_slope

                processor = RegressionGroupProcessing(self._group_params)
                with group_file_group_context_slope(self._all_results) as group:
                    result = processor.process_group(group)
            else:
                from bidsforge.processing.trial_stats_group import (
                    ConditionTestGroupProcessing,
                )

                from ._bridge import group_file_group_context

                processor = ConditionTestGroupProcessing(self._group_params)
                with group_file_group_context(
                    self._all_results,
                    primary_condition_metric=self._group_params.primary_condition_metric,
                ) as group:
                    result = processor.process_group(group)
            self.result_ready.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class WriteAllWorker(QThread):
    """Write all per-subject ``ConditionTestProcessingResult`` objects in a background thread.

    Parameters
    ----------
    results:
        Mapping of subject id to ``ConditionTestProcessingResult``.
    writer:
        A fully configured ``ConditionTestProcessingWriter`` instance.

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
        results: "dict[str, ConditionTestProcessingResult]",
        writer: "ConditionTestProcessingWriter",
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
    """Write a single ``ConditionTestGroupProcessingResult`` in a background thread.

    Parameters
    ----------
    result:
        The group-level result to write.
    writer:
        A fully configured ``ConditionTestGroupProcessingWriter`` instance.

    Signals
    -------
    finished : emitted on success.
    error    : str  — emitted with the exception message on failure.
    """

    finished = Signal()
    error = Signal(str)

    def __init__(
        self,
        result: "ConditionTestGroupProcessingResult",
        writer: "ConditionTestGroupProcessingWriter",
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
    by ``ConditionTestProcessingWriter`` and reconstructs ``ConditionTestProcessingResult``
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

    subject_done = Signal(str, object)  # subject_id, ConditionTestProcessingResult
    all_done = Signal(object)           # dict[str, ConditionTestProcessingResult]
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
        from bidsforge.processing.trial_stats import (
            load_regression_result,
        )
        from bidsforge.processing.trial_stats import (
            load_condition_test_result,
        )

        results: dict[
            str, "ConditionTestProcessingResult | RegressionProcessingResult"
        ] = {}
        total = len(self._subject_files)
        try:
            for i, (subject_id, path) in enumerate(self._subject_files.items(), start=1):
                self.progress.emit(
                    f"Loading subject {subject_id}  ({i}/{total})…"
                )
                result = _load_subject_result_auto(
                    path,
                    load_condition_test_result=load_condition_test_result,
                    load_regression_result=load_regression_result,
                )
                results[subject_id] = result
                self.subject_done.emit(subject_id, result)
            self.all_done.emit(results)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


class LoadGroupResultWorker(QThread):
    """Load a pre-computed group result (condition-test or regression) from a file.

    Parameters
    ----------
    group_file:
        Path to the ``.h5``/``.hdf5`` or ``.mat`` group stats file written by
        ``ConditionTestGroupProcessingWriter`` or
        ``RegressionGroupProcessingWriter``.

    Signals
    -------
    result_ready : emitted with the loaded group result on success.
    error        : str — emitted with the exception message on failure.
    """

    result_ready = Signal(object)  # ConditionTestGroupProcessingResult
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
            from bidsforge.processing.trial_stats_group import (
                load_regression_group_result,
            )
            from bidsforge.processing.trial_stats_group import (
                load_condition_test_group_result,
            )

            result = _load_group_result_auto(
                self._group_file,
                load_condition_test_group_result=load_condition_test_group_result,
                load_regression_group_result=load_regression_group_result,
            )
            self.result_ready.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.error.emit(str(exc))


def _load_subject_result_auto(
    path: Path,
    *,
    load_condition_test_result,
    load_regression_result,
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
                return load_regression_result(path)
            return load_condition_test_result(path)
        except Exception:
            # Fallback: try slope loader first, then classic trial-stats.
            try:
                return load_regression_result(path)
            except Exception:
                return load_condition_test_result(path)

    # MATLAB and others: optimistic slope-first fallback strategy.
    try:
        return load_regression_result(path)
    except Exception:
        return load_condition_test_result(path)


def _load_group_result_auto(
    path: Path,
    *,
    load_condition_test_group_result,
    load_regression_group_result,
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
                if analysis_type == "regression_group" or "regression" in fh.get("stats", {}):
                    return load_regression_group_result(path)

                if analysis_type == "condition_test_group" or "signal_activity" in fh.get("stats", {}):
                    return load_condition_test_group_result(path)
        except Exception:
            pass

    try:
        return load_condition_test_group_result(path)
    except Exception:
        return load_regression_group_result(path)


def _is_slope_group_params(group_params: object) -> bool:
    from bidsforge.processing.trial_stats_group import (
        RegressionGroupParams,
    )

    return isinstance(group_params, RegressionGroupParams)



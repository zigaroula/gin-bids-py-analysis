from __future__ import annotations

import json
import logging
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Lock
from typing import Any

logger = logging.getLogger(__name__)
_LOGGER_FALLBACK_SETUP_LOCK = Lock()

from joblib import Parallel, delayed, effective_n_jobs
from pydantic import BaseModel, computed_field
import multiprocessing as mp

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.file_group import BIDSFileGroup
from bidsforge.bids.helpers import build_bids_path

_PROVENANCE_ENTITIES = frozenset({"suffix", "extension", "datatype", "desc", "description"})


def _has_non_null_handler(candidate: logging.Logger) -> bool:
    """Return True when candidate or an ancestor has a real output handler."""
    current = candidate
    while current is not None:
        for handler in current.handlers:
            if not isinstance(handler, logging.NullHandler):
                return True
        if not current.propagate:
            break
        current = current.parent
    return False


def _ensure_logger_output() -> None:
    """Install a fallback stderr handler when no logging output is configured."""
    with _LOGGER_FALLBACK_SETUP_LOCK:
        if _has_non_null_handler(logger):
            return
        if any(h.get_name() == "bidsforge_fallback" for h in logger.handlers):
            return
        handler = logging.StreamHandler()
        handler.set_name("bidsforge_fallback")
        handler.setFormatter(
            logging.Formatter("%(levelname)s:%(name)s:%(message)s")
        )
        logger.addHandler(handler)

def _coerce_to_groups(items: list[BIDSFile | BIDSFileGroup]) -> list[BIDSFileGroup]:
    """Wrap bare BIDSFile objects into single-file BIDSFileGroups."""
    return [
        BIDSFileGroup(primary=item) if isinstance(item, BIDSFile) else item
        for item in items
    ]

class BaseProcessingParams(BaseModel):
    """
    Base class for all processing parameter models.

    Subclass this in each analysis subpackage to declare algorithm parameters
    (e.g. filter settings, frequency bands, sampling rate).  An instance is
    passed to the processor constructor and stored as ``self.params`` for use
    in :meth:`process_group`.
    """


class BaseWriterParams(BaseModel):
    """
    Base class for all writer parameter models.

    Carries the output routing fields required by
    :meth:`BaseProcessingWriter.write` plus any format-specific options declared
    in subclasses (e.g. output format, compression level).

    Required fields (must be provided by caller or overridden with defaults
    in a concrete subclass):

    - ``bids_root`` — root of the BIDS dataset; ``derivatives/<pipeline_label>``
      is created automatically.
    - ``pipeline_label`` — becomes ``desc-<label>`` in the output filename.
    - ``output_modality`` — BIDS modality of the output file.
    - ``output_suffix`` — BIDS suffix of the output file.
    - ``output_format`` — human-readable format name declared by each concrete
      subclass (e.g. ``"hdf5"`` or ``"brainvision"``).  Concrete subclasses
      can override :attr:`output_extension` as a ``@computed_field`` that
      maps their allowed format names to file extensions.
    """

    bids_root: Path
    pipeline_label: str
    output_modality: str
    output_suffix: str
    output_description: str
    output_format: str


    @computed_field
    @property
    def output_extension(self) -> str:
        """File extension derived from ``output_format``.  Can be overridden by subclasses."""
        if self.output_format == "hdf5":
            return ".h5"
        elif self.output_format == "tsv":
            return ".tsv"
        elif self.output_format == "brainvision":
            return ".vhdr"
        elif self.output_format == "matlab":
            return ".mat"
        else:
            raise ValueError(f"Unsupported output_format: {self.output_format}")


@dataclass
class BaseProcessingResult(ABC):
    """
    Abstract base for all processing results.

    Each analysis defines its own concrete subclass carrying the specific
    output data (e.g. numpy arrays, frequency-band envelopes) alongside
    the provenance fields defined here.

    Attributes:
        source_group:   The :class:`~bidsforge.bids.file_group.BIDSFileGroup`
                        that was processed.  ``source_group.primary`` is used by
                        writers to construct the BIDS output path.
        metadata:       Arbitrary key/value pairs for algorithm-level metadata
                        (e.g. sampling frequency, filter parameters used).
    """

    source_group: BIDSFileGroup
    metadata: dict[str, Any] = field(default_factory=dict)
    output_entities: dict[str, Any] | None = None


class BaseProcessingWriter(ABC):
    """
    Abstract base for all BIDS-derivatives writers.

    Uses the **Template Method** pattern:

    - :meth:`write` is **concrete** and handles all shared setup: it derives
      the output path from ``result.source_group.primary``'s entities plus the
      pipeline label (taken from ``params.pipeline_label``), creates the output
      directory, and then delegates the actual serialisation to
      :meth:`_write_data`.
    - :meth:`_write_data` is **abstract** and is the only method subclasses
      need to implement.

    The constructor takes a :class:`BaseWriterParams` instance (or a subclass)
    that provides the output routing config (``bids_root``, ``pipeline_label``,
    ``output_suffix``, ``output_extension``) plus any format-specific options
    declared by the concrete subclass.
    """

    def __init__(self, params: BaseWriterParams) -> None:
        self.params = params if params is not None else BaseWriterParams()

    def _build_output_path(self, entities: dict[str, Any]) -> Path:
        """Build the BIDS derivative output path from a pre-resolved entity dict.

        Strips provenance-only keys, injects ``desc``, and delegates to
        :func:`~bidsforge.bids.helpers.build_bids_path`.  This is the
        single source of truth for path construction shared by :meth:`write` and
        :meth:`get_output_path`.

        Args:
            entities: Raw entity dict (e.g. from ``BIDSFile.entities`` or
                      ``result.output_entities``).  Provenance keys such as
                      ``suffix`` and ``extension`` are stripped automatically.

        Returns:
            The fully resolved BIDS derivative :class:`~pathlib.Path`.
        """
        output_root = self.params.bids_root / "derivatives" / self.params.pipeline_label
        clean = {k: v for k, v in entities.items() if k not in _PROVENANCE_ENTITIES}
        clean["desc"] = self.params.output_description
        return build_bids_path(
            entities=clean,
            root=output_root,
            suffix=self.params.output_suffix,
            extension=self.params.output_extension,
            datatype=self.params.output_modality,
        )

    def write(
        self,
        result: BaseProcessingResult,
    ) -> Path:
        """
        Construct the BIDS output path, create directories, then write *result*.

        The output path is derived from ``result.source_group.primary``'s
        entities plus ``desc-<OUTPUT_DESCRIPTION>``, rooted at
        ``bids_root / "derivatives" / PIPELINE_LABEL``.  Subclasses never need
        to override this method - implement :meth:`_write_data` instead.

        Returns:
            :class:`~pathlib.Path` to the written output file.
        """
        source_entities = (
            result.output_entities
            if result.output_entities is not None
            else result.source_group.primary.entities
        )
        output_path = self._build_output_path(source_entities)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._write_data(result, output_path)
        return output_path

    def get_output_path(self, group: BIDSFileGroup) -> Path:
        """Compute the expected output path for *group* without processing it.

        Used by :meth:`BaseProcessing.run` when ``skip_existing=True`` to check
        whether a result is already on disk before starting computation.

        The path is built from ``group.primary.entities`` using the same rules
        as :meth:`write`.  Subclasses whose processor sets
        ``result.output_entities`` should override this method to mirror that
        override so the predicted path is accurate.

        Args:
            group: The file group for which to predict the output path.

        Returns:
            The predicted BIDS derivative output path.
        """
        return self._build_output_path(group.primary.entities)

    def write_dataset_description(self, processing_params: BaseProcessingParams | None = None) -> None:
        """
        Write (or overwrite) a BIDS-compliant ``dataset_description.json`` at
        ``bids_root / "derivatives" / pipeline_label``.

        Called once at the start of :meth:`BaseProcessing.run` before the
        processing loop.  Subclasses never need to override this.

        Args:
            processing_params: The algorithm parameters used for this run
                (a :class:`BaseProcessingParams` instance).  When provided,
                the full parameter dict is written under
                ``GeneratedBy[0]["Parameters"]``.  Pass ``None`` to omit it.
        """
        derivatives_root = self.params.bids_root / "derivatives" / self.params.pipeline_label
        derivatives_root.mkdir(parents=True, exist_ok=True)

        try:
            pkg_version = version(self.params.pipeline_label)
        except PackageNotFoundError:
            pkg_version = "unknown"

        generated_by: dict[str, Any] = {
            "Name": self.params.pipeline_label,
            "Version": pkg_version,
        }
        if processing_params is not None:
            generated_by["Parameters"] = processing_params.model_dump()

        description = {
            "Name": self.params.pipeline_label,
            "BIDSVersion": "1.11.1",
            "DatasetType": "derivative",
            "GeneratedBy": [generated_by],
        }

        dest = derivatives_root / "dataset_description.json"
        dest.write_text(json.dumps(description, indent=2), encoding="utf-8")

    @abstractmethod
    def _write_data(self, result: BaseProcessingResult, output_path: Path) -> None:
        """
        Serialise *result* to *output_path*.

        The output directory is guaranteed to exist when this method is called.
        This is the only method subclasses need to implement.

        Args:
            result:      The in-memory processing result to serialise.
            output_path: Fully resolved, BIDS-compliant destination path.
        """


class BaseProcessing(ABC):
    """
    Abstract base for all analysis processors.

    Subclasses implement :meth:`process_group`, which handles one
    :class:`~bidsforge.bids.file_group.BIDSFileGroup` at a time.
    :meth:`execute` and :meth:`run` are fully implemented here and fan out
    to :meth:`process_group` with optional joblib parallelism.

    A processor is **stateless with respect to file I/O**: results are returned
    in-memory.  Writing to disk is handled by a :class:`BaseProcessingWriter`
    subclass.
    """

    @abstractmethod
    def process_group(self, group: BIDSFileGroup, progress_tracking_position: int = 0) -> BaseProcessingResult:
        """
        Process a single :class:`~bidsforge.bids.file_group.BIDSFileGroup`
        and return a result object.

        This is the only method subclasses need to implement.  For single-file
        analyses the group will contain just a primary file.  For multi-modal
        analyses it will also carry secondary files (e.g. a physio recording
        alongside the primary iEEG file).

        Args:
            group: The file group to process.
            progress_tracking_position: Optional position index for progress tracking (e.g. with tqdm).

        Returns:
            An instance of a :class:`BaseProcessingResult` subclass.
        """
    
    def process_file(self, file: BIDSFile, progress_tracking_position: int = 0) -> BaseProcessingResult:
        """
        Convenience method to process a single :class:`~bidsforge.bids.file.BIDSFile`.

        Wraps the file in a single-file :class:`~bidsforge.bids.file_group.BIDSFileGroup`
        and delegates to :meth:`process_group`.  This is useful for simple analyses
        that don't need multi-file groups.

        Args:
            file: The file to process.
            progress_tracking_position: Optional position index for progress tracking (passed to process_group).
        
        Returns:
            An instance of a :class:`BaseProcessingResult` subclass.
        """
        group = BIDSFileGroup(primary=file)
        return self.process_group(group, progress_tracking_position=progress_tracking_position)

    def execute(
        self,
        groups: list[BIDSFile | BIDSFileGroup],
        n_jobs: int = 1,
    ) -> list[BaseProcessingResult]:
        """
        Run the analysis on each group and return all results in memory.

        Parallelism is handled internally via joblib.  The processor must be
        stateless - no shared mutable state across calls to :meth:`process_group`.

        Args:
            groups: :class:`~bidsforge.bids.file_group.BIDSFileGroup`
                    or bare :class:`~bidsforge.bids.file.BIDSFile`
                    objects (auto-wrapped into single-file groups).
            n_jobs: Number of parallel workers (``1`` = sequential, ``-1`` = all CPUs).

        Returns:
            A list of :class:`BaseProcessingResult` subclass instances,
            one per input group, in the same order as *groups*.
        """
        _ensure_logger_output()
        coerced = _coerce_to_groups(groups)
        resolved_n_jobs = max(1, effective_n_jobs(n_jobs))

        # Avoid multiprocessing.Manager() entirely for effectively sequential
        # runs. On Windows, creating a Manager can stall while spawning helper
        # processes even when n_jobs=1, which makes simple one-file scripts
        # appear hung before any real processing begins.
        if resolved_n_jobs == 1:
            results: list[BaseProcessingResult] = []
            for g in coerced:
                try:
                    result = self.process_group(g, progress_tracking_position=0)
                except Exception:
                    logger.error(
                        "Error processing group %s â€” skipping.\n%s",
                        g.primary.path,
                        traceback.format_exc(),
                    )
                    continue
                results.append(result)
            return results

        # Set up a multiprocessing-safe queue to assign worker positions for
        # progress tracking in true parallel runs only.
        manager = mp.Manager()
        position_queue = manager.Queue()
        for pos in range(resolved_n_jobs):
            position_queue.put(pos)

        def _process(g: BIDSFileGroup) -> BaseProcessingResult | None:
            pos = position_queue.get()  # Get a position for this worker
            try:
                return self.process_group(g, progress_tracking_position=pos)
            except Exception:
                logger.error(
                    "Error processing group %s — skipping.\n%s",
                    g.primary.path,
                    traceback.format_exc(),
                )
                return None
            finally:
                position_queue.put(pos)  # Return the position to the queue

        results = Parallel(n_jobs=n_jobs)(delayed(_process)(g) for g in coerced)
        return [r for r in results if r is not None]

    def run(
        self,
        groups: list[BIDSFile | BIDSFileGroup],
        writer: "BaseProcessingWriter",
        n_jobs: int = 1,
        skip_existing: bool = False,
    ) -> list[Path]:
        """
        Process each group and write its result immediately, then discard it
        from memory.  This is the recommended entry-point for scripts.

        Unlike :meth:`execute`, results are never accumulated in memory: each
        worker processes one group, writes it, and frees the result before
        moving on.

        Args:
            groups:        :class:`~bidsforge.bids.file_group.BIDSFileGroup`
                           or bare :class:`~bidsforge.bids.file.BIDSFile`
                           objects (auto-wrapped into single-file groups).
            writer:        Writer instance (holds the BIDS root, derives the
                           derivatives subfolder automatically).
            n_jobs:        Parallel workers (``1`` = sequential, ``-1`` = all CPUs).
            skip_existing: When ``True``, groups whose predicted output path
                           already exists on disk are skipped without processing.
                           Their existing paths are included in the return value.
                           Use this to resume an interrupted run without
                           reprocessing already-completed groups.

        Returns:
            List of output :class:`~pathlib.Path` objects (both newly written
            and, when *skip_existing* is ``True``, previously existing ones).
        """
        _ensure_logger_output()
        writer.write_dataset_description(getattr(self, "params", None))
        coerced = _coerce_to_groups(groups)
        resolved_n_jobs = max(1, effective_n_jobs(n_jobs))

        # Keep n_jobs=1 truly sequential so Windows callers do not pay the
        # cost of spinning up a multiprocessing.Manager() server process.
        if resolved_n_jobs == 1:
            paths: list[Path] = []
            for g in coerced:
                if skip_existing:
                    expected_path = writer.get_output_path(g)
                    if expected_path.exists():
                        logger.info(
                            "Skipping %s \u2014 output already exists at %s",
                            g.primary.path,
                            expected_path,
                        )
                        paths.append(expected_path)
                        continue
                try:
                    result = self.process_group(g, progress_tracking_position=0)
                    out_path = writer.write(result)
                except Exception:
                    logger.error(
                        "Error processing group %s \u2014 skipping.\n%s",
                        g.primary.path,
                        traceback.format_exc(),
                    )
                    continue
                paths.append(out_path)
            return paths

        # Parallel path: partition groups into those already done and those
        # that still need processing before spinning up workers.
        already_done: list[Path] = []
        if skip_existing:
            remaining: list[BIDSFileGroup] = []
            for g in coerced:
                expected_path = writer.get_output_path(g)
                if expected_path.exists():
                    logger.info(
                        "Skipping %s \u2014 output already exists at %s",
                        g.primary.path,
                        expected_path,
                    )
                    already_done.append(expected_path)
                else:
                    remaining.append(g)
            coerced = remaining

        if not coerced:
            return already_done

        # Set up a multiprocessing-safe queue to assign worker positions for
        # progress tracking in true parallel runs only.
        manager = mp.Manager()
        position_queue = manager.Queue()
        for pos in range(resolved_n_jobs):
            position_queue.put(pos)

        def _process_and_write(g: BIDSFileGroup) -> Path | None:
            pos = position_queue.get()  # Get a position for this worker
            try:
                result = self.process_group(g, progress_tracking_position=pos)
                return writer.write(result)
            except Exception:
                logger.error(
                    "Error processing group %s \u2014 skipping.\n%s",
                    g.primary.path,
                    traceback.format_exc(),
                )
                return None
            finally:
                position_queue.put(pos)  # Return the position to the queue

        new_paths = Parallel(n_jobs=n_jobs)(delayed(_process_and_write)(g) for g in coerced)
        return already_done + [p for p in new_paths if p is not None]

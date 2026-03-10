from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from joblib import Parallel, delayed

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.bids.helpers import build_bids_path

_PROVENANCE_ENTITIES = frozenset({"suffix", "extension", "datatype"})

def _coerce_to_groups(items: list[BIDSFile | BIDSFileGroup]) -> list[BIDSFileGroup]:
    """Wrap bare BIDSFile objects into single-file BIDSFileGroups."""
    return [
        BIDSFileGroup(primary=item) if isinstance(item, BIDSFile) else item
        for item in items
    ]


@dataclass
class BaseProcessingResult(ABC):
    """
    Abstract base for all processing results.

    Each analysis defines its own concrete subclass carrying the specific
    output data (e.g. numpy arrays, frequency-band envelopes) alongside
    the provenance fields defined here.

    Attributes:
        source_group:   The :class:`~gin_bids_py_analysis.bids.file_group.BIDSFileGroup`
                        that was processed.  ``source_group.primary`` is used by
                        writers to construct the BIDS output path.
        metadata:       Arbitrary key/value pairs for algorithm-level metadata
                        (e.g. sampling frequency, filter parameters used).
    """

    source_group: BIDSFileGroup
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseProcessingWriter(ABC):
    """
    Abstract base for all BIDS-derivatives writers.

    Uses the **Template Method** pattern:

    - :meth:`write` is **concrete** and handles all shared setup: it derives
      the output path from ``result.source_group.primary``'s entities plus the
      pipeline label, creates the output directory, and then delegates the
      actual serialisation to :meth:`_write_data`.
    - :meth:`_write_data` is **abstract** and is the only method subclasses
      need to implement.

    Subclasses must also declare three class-level constants:

    .. code-block:: python

        PIPELINE_LABEL = "hilbert"   # becomes desc-<label> in the filename
        OUTPUT_SUFFIX  = "hilbert"   # BIDS suffix of the output file
        OUTPUT_EXTENSION = ".npy"    # file extension including the dot
    """

    PIPELINE_LABEL: str
    OUTPUT_SUFFIX: str
    OUTPUT_EXTENSION: str

    def __init__(self, bids_root: Path) -> None:
        self._bids_root = bids_root

    def write(
        self,
        result: BaseProcessingResult,
    ) -> Path:
        """
        Construct the BIDS output path, create directories, then write *result*.

        The output path is derived from ``result.source_group.primary``'s
        entities plus ``desc-<PIPELINE_LABEL>``, rooted at
        ``bids_root / "derivatives" / PIPELINE_LABEL``.  Subclasses never need
        to override this method - implement :meth:`_write_data` instead.

        Returns:
            :class:`~pathlib.Path` to the written output file.
        """
        output_root = self._bids_root / "derivatives" / self.PIPELINE_LABEL
        primary = result.source_group.primary
        entities = {
            k: v
            for k, v in primary.entities.items()
            if k not in _PROVENANCE_ENTITIES
        }
        entities["desc"] = self.PIPELINE_LABEL

        output_path = build_bids_path(
            entities=entities,
            root=output_root,
            suffix=self.OUTPUT_SUFFIX,
            extension=self.OUTPUT_EXTENSION,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._write_data(result, output_path)
        return output_path

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
    :class:`~gin_bids_py_analysis.bids.file_group.BIDSFileGroup` at a time.
    :meth:`execute` and :meth:`run` are fully implemented here and fan out
    to :meth:`process_group` with optional joblib parallelism.

    A processor is **stateless with respect to file I/O**: results are returned
    in-memory.  Writing to disk is handled by a :class:`BaseProcessingWriter`
    subclass.
    """

    @abstractmethod
    def process_group(self, group: BIDSFileGroup) -> BaseProcessingResult:
        """
        Process a single :class:`~gin_bids_py_analysis.bids.file_group.BIDSFileGroup`
        and return a result object.

        This is the only method subclasses need to implement.  For single-file
        analyses the group will contain just a primary file.  For multi-modal
        analyses it will also carry secondary files (e.g. a physio recording
        alongside the primary iEEG file).

        Args:
            group: The file group to process.

        Returns:
            An instance of a :class:`BaseProcessingResult` subclass.
        """
    
    def process_file(self, file: BIDSFile) -> BaseProcessingResult:
        """
        Convenience method to process a single :class:`~gin_bids_py_analysis.bids.file.BIDSFile`.

        Wraps the file in a single-file :class:`~gin_bids_py_analysis.bids.file_group.BIDSFileGroup`
        and delegates to :meth:`process_group`.  This is useful for simple analyses
        that don't need multi-file groups.

        Args:
            file: The file to process.
        
        Returns:
            An instance of a :class:`BaseProcessingResult` subclass.
        """
        group = BIDSFileGroup(primary=file)
        return self.process_group(group)

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
            groups: :class:`~gin_bids_py_analysis.bids.file_group.BIDSFileGroup`
                    or bare :class:`~gin_bids_py_analysis.bids.file.BIDSFile`
                    objects (auto-wrapped into single-file groups).
            n_jobs: Number of parallel workers (``1`` = sequential, ``-1`` = all CPUs).

        Returns:
            A list of :class:`BaseProcessingResult` subclass instances,
            one per input group, in the same order as *groups*.
        """
        coerced = _coerce_to_groups(groups)
        return Parallel(n_jobs=n_jobs)(delayed(self.process_group)(g) for g in coerced)

    def run(
        self,
        groups: list[BIDSFile | BIDSFileGroup],
        writer: "BaseProcessingWriter",
        n_jobs: int = 1,
    ) -> list[Path]:
        """
        Process each group and write its result immediately, then discard it
        from memory.  This is the recommended entry-point for scripts.

        Unlike :meth:`execute`, results are never accumulated in memory: each
        worker processes one group, writes it, and frees the result before
        moving on.

        Args:
            groups:  :class:`~gin_bids_py_analysis.bids.file_group.BIDSFileGroup`
                     or bare :class:`~gin_bids_py_analysis.bids.file.BIDSFile`
                     objects (auto-wrapped into single-file groups).
            writer:  Writer instance (holds the BIDS root, derives the
                     derivatives subfolder automatically).
            n_jobs:  Parallel workers (``1`` = sequential, ``-1`` = all CPUs).

        Returns:
            List of output :class:`~pathlib.Path` objects in group order.
        """
        coerced = _coerce_to_groups(groups)

        def _process_and_write(g: BIDSFileGroup) -> Path:
            result = self.process_group(g)
            return writer.write(result)

        return Parallel(n_jobs=n_jobs)(delayed(_process_and_write)(g) for g in coerced)

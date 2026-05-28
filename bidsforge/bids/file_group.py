from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .file import BIDSFile

if TYPE_CHECKING:
    from .dataset import BIDSDataset


@dataclass
class BIDSFileGroup:
    """
    A group of related :class:`BIDSFile` objects to be processed together.

    One file is designated as *primary* — it drives BIDS output-path
    construction (i.e. the output filename is derived from its entities).
    All remaining files are *secondaries* (e.g. a physio channel recording
    accompanying the primary iEEG file).

    For single-file analyses simply omit *secondaries*::

        group = BIDSFileGroup(primary=ieeg_file)

    For multi-modal analyses supply the companion files explicitly::

        group = BIDSFileGroup(primary=ieeg_file, secondaries=[physio_file])

    Grouping is the caller's responsibility (typically done in run scripts),
    not the processor's.
    """

    primary: BIDSFile
    secondaries: list[BIDSFile] = field(default_factory=list)

    @property
    def all_files(self) -> list[BIDSFile]:
        """All files in this group: primary first, then secondaries."""
        return [self.primary] + self.secondaries

    def __repr__(self) -> str:
        n = len(self.secondaries)
        secondary_info = f", {n} secondary file(s)" if n else ""
        return f"BIDSFileGroup(primary={self.primary.path.name!r}{secondary_info})"


def build_subject_groups(
    dataset: "BIDSDataset",
    ieeg_filters: dict[str, Any],
    secondary_filters: list[dict[str, Any]],
    *,
    aggregate_runs: bool = True,
) -> list[BIDSFileGroup]:
    """Group BIDS files into one :class:`BIDSFileGroup` per subject (or per
    recording) ready to be passed to a processor.

    The function queries *dataset* with *ieeg_filters* to find primary iEEG
    files, and with each entry in *secondary_filters* to find companion files
    (behaviour tables, electrodes tables, etc.).  Secondary files are matched
    to primary files by ``subject`` (and ``session`` when not aggregating runs).

    Parameters
    ----------
    dataset:
        A :class:`~bidsforge.bids.BIDSDataset` already loaded.
    ieeg_filters:
        Entity filters forwarded to
        :meth:`~bidsforge.bids.BIDSDataset.get_files` to select
        the primary iEEG files (e.g. ``{"suffix": "ieeg", "extension": ".vhdr",
        "desc": "gammasm0"}``).
    secondary_filters:
        List of entity-filter dicts, each forwarded to
        :meth:`~bidsforge.bids.BIDSDataset.get_files` to discover
        companion files.  Duplicate paths across entries are de-duplicated.
    aggregate_runs:
        When ``True`` (default), all recordings that share the same ``subject``
        are pooled into a single group.  The first recording (sorted by path)
        becomes the ``primary``; all others plus all secondary files for that
        subject become ``secondaries``.

        When ``False``, each iEEG file gets its own group.  Secondary files
        are matched by ``subject`` and ``session`` (run-agnostic, because files
        such as ``*_electrodes.tsv`` are typically shared within a session).

    Returns
    -------
    list[BIDSFileGroup]
        One entry per subject when *aggregate_runs* is ``True``, or one entry
        per iEEG file when ``False``.  Groups are sorted by the path of their
        primary file.
    """
    all_ieeg: list[BIDSFile] = sorted(
        dataset.get_files(**ieeg_filters),
        key=lambda f: str(f.path),
    )

    # Collect secondary files, de-duplicated by path
    sec_by_path: dict[Path, BIDSFile] = {}
    for entity_filters in secondary_filters:
        for file in dataset.get_files(**entity_filters):
            sec_by_path[file.path] = file
    all_secondary: list[BIDSFile] = sorted(
        sec_by_path.values(), key=lambda f: str(f.path)
    )

    groups: list[BIDSFileGroup] = []

    if aggregate_runs:
        for subject_id in sorted({f.get("subject") for f in all_ieeg}):
            subject_ieeg = [f for f in all_ieeg if f.get("subject") == subject_id]
            if not subject_ieeg:
                continue
            subject_secondaries = [
                f for f in all_secondary if f.get("subject") == subject_id
            ]
            groups.append(
                BIDSFileGroup(
                    primary=subject_ieeg[0],
                    secondaries=subject_ieeg[1:] + subject_secondaries,
                )
            )
    else:
        for ieeg_file in all_ieeg:
            subj = ieeg_file.get("subject")
            ses = ieeg_file.get("session")
            file_secondaries = [
                f
                for f in all_secondary
                if f.get("subject") == subj
                and (
                    ses is None
                    or f.get("session") is None
                    or f.get("session") == ses
                )
            ]
            groups.append(
                BIDSFileGroup(primary=ieeg_file, secondaries=file_secondaries)
            )

    return groups

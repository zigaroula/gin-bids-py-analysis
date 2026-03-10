from __future__ import annotations

from dataclasses import dataclass, field

from .file import BIDSFile


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

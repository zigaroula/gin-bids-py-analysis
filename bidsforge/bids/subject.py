from __future__ import annotations

from typing import Any

from .file import BIDSFile


class BIDSSubject:
    """
    A BIDS participant, holding all associated :class:`BIDSFile` objects.

    Use :meth:`get_files` to filter files by any combination of BIDS entities.
    """

    def __init__(self, subject_id: str, files: list[BIDSFile]) -> None:
        self.subject_id = subject_id
        self._files = list(files)

    @property
    def files(self) -> list[BIDSFile]:
        """All files associated with this subject (read-only copy)."""
        return list(self._files)

    def get_files(self, **entities: Any) -> list[BIDSFile]:
        """
        Return files matching ALL supplied entity key=value filters.

        Any BIDS entity key is valid (e.g. ``suffix``, ``task``, ``run``, ``acq``).

        Example::

            subject.get_files(suffix="ieeg", run="1")
        """
        result = []
        for f in self._files:
            if all(f.get(k) == str(v) for k, v in entities.items()):
                result.append(f)
        return result

    def __repr__(self) -> str:
        return f"BIDSSubject(id={self.subject_id!r}, n_files={len(self._files)})"

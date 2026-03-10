from __future__ import annotations

from pathlib import Path
from typing import Any

import bids.layout as _layout


class BIDSFile:
    """
    Thin wrapper around a pybids BIDSFile exposing a uniform entity interface.

    All BIDS entities are accessible via:
    - ``file.entities``        — full dict of all parsed entities
    - ``file['subject']``      — subscript access (raises KeyError if missing)
    - ``file.get('run')``      — safe access with optional default
    - ``file.suffix``          — convenience property
    - ``file.extension``       — convenience property
    - ``file.datatype``        — convenience property
    - ``file.path``            — ``pathlib.Path`` to the file
    """

    def __init__(self, pybids_file: _layout.BIDSFile) -> None:
        self._file = pybids_file

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return Path(self._file.path)

    @property
    def entities(self) -> dict[str, Any]:
        """All parsed BIDS entities as a plain dict (e.g. ``{'subject': '01', 'suffix': 'ieeg', ...}``)."""
        return dict(self._file.entities)

    @property
    def suffix(self) -> str | None:
        return self._file.entities.get("suffix")

    @property
    def extension(self) -> str | None:
        return self._file.entities.get("extension")

    @property
    def datatype(self) -> str | None:
        return self._file.entities.get("datatype")

    # ------------------------------------------------------------------
    # Dynamic entity access
    # ------------------------------------------------------------------

    def __getitem__(self, entity: str) -> Any:
        """Allow ``file['subject']``, ``file['acq']``, ``file['run']`` etc."""
        try:
            return self._file.entities[entity]
        except KeyError:
            raise KeyError(f"Entity {entity!r} not found in {self.path.name!r}") from None

    def get(self, entity: str, default: Any = None) -> Any:
        """Return the value for *entity* or *default* if not present."""
        return self._file.entities.get(entity, default)

    def __repr__(self) -> str:
        return f"BIDSFile({self.path.name!r})"

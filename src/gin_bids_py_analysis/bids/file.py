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
        # Eagerly resolve all data from the SQLAlchemy-backed pybids object
        # while we are in the main process and the session is alive.
        # This makes BIDSFile fully pickle-safe for joblib multiprocessing.
        self._path = Path(pybids_file.path)
        self._entities: dict[str, Any] = dict(pybids_file.entities)

    @classmethod
    def from_path(
        cls,
        path: Path | str,
    ) -> "BIDSFile":
        """
        Alternate constructor for non-pybids contexts.

        Use this when only a path is available (outside a BIDSDataset query).
        """
        from .helpers import parse_entities

        resolved_path = Path(path)
        entities = parse_entities(resolved_path)
        if "sub" in entities and "subject" not in entities:
            entities["subject"] = entities["sub"]
        if "ses" in entities and "session" not in entities:
            entities["session"] = entities["ses"]
        entities["suffix"] = resolved_path.stem.split("_")[-1]
        entities["extension"] = resolved_path.suffix
        entities["datatype"] = resolved_path.parent.name

        instance = cls.__new__(cls)
        instance._path = resolved_path
        instance._entities = dict(entities)
        return instance

    # ------------------------------------------------------------------
    # Core properties
    # ------------------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._path

    @property
    def entities(self) -> dict[str, Any]:
        """All parsed BIDS entities as a plain dict (e.g. ``{'subject': '01', 'suffix': 'ieeg', ...}``)."""
        return dict(self._entities)

    @property
    def suffix(self) -> str | None:
        return self._entities.get("suffix")

    @property
    def extension(self) -> str | None:
        return self._entities.get("extension")

    @property
    def datatype(self) -> str | None:
        return self._entities.get("datatype")

    # ------------------------------------------------------------------
    # Dynamic entity access
    # ------------------------------------------------------------------

    def __getitem__(self, entity: str) -> Any:
        """Allow ``file['subject']``, ``file['acq']``, ``file['run']`` etc."""
        try:
            return self._entities[entity]
        except KeyError:
            raise KeyError(f"Entity {entity!r} not found in {self._path.name!r}") from None

    def get(self, entity: str, default: Any = None) -> Any:
        """Return the value for *entity* or *default* if not present."""
        return self._entities.get(entity, default)

    def __repr__(self) -> str:
        return f"BIDSFile({self.path.name!r})"

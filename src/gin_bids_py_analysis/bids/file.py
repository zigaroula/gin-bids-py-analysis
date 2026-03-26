from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import bids.layout as _layout

# Maps file extensions to the name of the loader function in data.loader that
# handles them.  Used by ensure_loaded() when no explicit loader is set.
_AUTO_DETECT_EXT_TO_LOADER: dict[str, str] = {
    ".vhdr": "load_ieeg",
    ".edf": "load_ieeg",
    ".bdf": "load_ieeg",
    ".fif": "load_ieeg",
    ".set": "load_ieeg",
    ".tsv": "load_table",
    ".csv": "load_table",
    ".json": "load_json",
    ".h5": "load_hdf5",
    ".hdf5": "load_hdf5",
    ".mat": "load_mat",
}


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
        self._data: Any = None
        self._loader: Callable[..., Any] | None = None
        self._load_kwargs: dict[str, Any] = {}
        self._externally_loaded: bool = False

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
        instance._data = None
        instance._loader = None
        instance._load_kwargs = {}
        instance._externally_loaded = False
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

    # ------------------------------------------------------------------
    # Loaded-data encapsulation
    # ------------------------------------------------------------------

    @property
    def data(self) -> Any:
        """Return the currently attached data object, or ``None`` if not loaded."""
        return self._data

    @property
    def is_loaded(self) -> bool:
        """Return ``True`` if a data object is currently attached to this file."""
        return self._data is not None

    def attach_data(self, data: Any) -> "BIDSFile":
        """Attach an already-loaded data object to this file.

        The attached data is returned as-is by :meth:`ensure_loaded` and is
        **not** discarded when the context exits — the caller retains ownership.

        Args:
            data: Any loaded object — :class:`mne.io.BaseRaw`,
                  ``list[dict[str, str]]``, ``dict``, or any domain-specific
                  object.

        Returns:
            ``self`` for method chaining.
        """
        self._data = data
        self._externally_loaded = True
        return self

    def preload(self, **kwargs: Any) -> "BIDSFile":
        """Load data from disk and attach it permanently.

        After calling this method, :meth:`ensure_loaded` will return the
        pre-loaded data without re-reading from disk on subsequent calls.
        The data persists because :meth:`ensure_loaded` takes its early-return
        path (``if self._data is not None``) and never reaches the ``finally``
        block that would otherwise clear it.

        Calling this method when data is already attached is a no-op.

        Args:
            **kwargs: Forwarded to the loader, merged with any defaults
                      registered via :meth:`set_loader`.

        Returns:
            ``self`` for method chaining.
        """
        if self._data is not None:
            return self
        loader = self._resolve_loader()
        merged_kwargs = {**self._load_kwargs, **kwargs}
        data = loader(self._path, **merged_kwargs)
        self.attach_data(data)
        return self

    def set_loader(self, fn: Callable[..., Any], **defaults: Any) -> "BIDSFile":
        """Override the automatic loader for this file.

        Use this when extension-based auto-detection is insufficient or when a
        domain-specific loader (e.g. a private HDF5 parser) should be used.

        Args:
            fn:        A callable with signature
                       ``(path: Path | str, **kwargs) -> Any``.
                       Must be a module-level function to survive joblib
                       pickling — lambdas are **not** supported.
            **defaults: Default keyword arguments forwarded to *fn* on every
                        call.  These are merged with (and overridden by) any
                        kwargs passed directly to :meth:`ensure_loaded`.

        Returns:
            ``self`` for method chaining.
        """
        self._loader = fn
        self._load_kwargs = {**self._load_kwargs, **defaults}
        return self

    @contextmanager
    def ensure_loaded(self, **kwargs: Any) -> Generator[Any, None, None]:
        """Context manager that yields the loaded data for this file.

        Two behaviours depending on state:

        * **Already loaded** (data was set via :meth:`attach_data`):
          yields the existing data object and does nothing on exit — the caller
          retains ownership and the data remains attached.

        * **Not yet loaded**:
          resolves the loader (explicit via :meth:`set_loader`, or
          auto-detected from the file extension), loads the data, yields it,
          then discards it on exit to reclaim memory.

        Auto-detected loaders by extension:

        +---------------------------+-------------+
        | Extension                 | Loader      |
        +===========================+=============+
        | ``.vhdr``, ``.edf``,      | load_ieeg   |
        | ``.bdf``, ``.fif``,       |             |
        | ``.set``                  |             |
        +---------------------------+-------------+
        | ``.tsv``, ``.csv``        | load_table  |
        +---------------------------+-------------+
        | ``.json``                 | load_json   |
        +---------------------------+-------------+
        | ``.h5``, ``.hdf5``        | load_hdf5   |
        +---------------------------+-------------+
        | ``.mat``                  | load_mat    |
        +---------------------------+-------------+

        Args:
            **kwargs: Forwarded to the loader, merged with any defaults
                      registered via :meth:`set_loader`.  These take
                      precedence over registered defaults.

        Yields:
            The loaded data object (type depends on the loader used).

        Raises:
            ValueError: If no loader is registered and the extension is not
                in the auto-detection table.

        Note:
            **Thread safety**: concurrent calls to ``ensure_loaded()`` on the
            same unloaded instance from multiple threads may race.  This is
            not a concern for joblib's default process-based parallelism, but
            should be avoided in threaded contexts.
        """
        if self._data is not None:
            yield self._data
            return

        loader = self._resolve_loader()
        merged_kwargs = {**self._load_kwargs, **kwargs}
        data = loader(self._path, **merged_kwargs)
        self._data = data
        try:
            yield data
        finally:
            if hasattr(data, "close") and callable(data.close):
                data.close()
            self._data = None

    def _resolve_loader(self) -> Callable[..., Any]:
        """Return the loader callable, either explicit or auto-detected."""
        if self._loader is not None:
            return self._loader

        ext = self._entities.get("extension", "")
        loader_name = _AUTO_DETECT_EXT_TO_LOADER.get(ext)
        if loader_name is None:
            raise ValueError(
                f"No loader registered for extension {ext!r} on "
                f"{self._path.name!r}. Use set_loader() to specify one "
                "explicitly."
            )
        # Lazy import: keeps bids/ independent of data/ at module-load time.
        from gin_bids_py_analysis.data.loader import (  # noqa: PLC0415
            load_hdf5,
            load_ieeg,
            load_json,
            load_mat,
            load_table,
        )

        _map: dict[str, Callable[..., Any]] = {
            "load_ieeg": load_ieeg,
            "load_table": load_table,
            "load_json": load_json,
            "load_hdf5": load_hdf5,
            "load_mat": load_mat,
        }
        return _map[loader_name]

    # ------------------------------------------------------------------
    # Pickle safety
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        """Exclude loaded data when pickling (e.g. for joblib workers).

        ``_data`` cannot generally survive pickling (MNE Raw objects are not
        picklable) and must not be sent to worker processes.
        ``_externally_loaded`` is reset so workers always load on demand via
        :meth:`ensure_loaded`.  ``_loader`` and ``_load_kwargs`` survive so
        workers can reconstruct the load call.
        """
        state = self.__dict__.copy()
        state["_data"] = None
        state["_externally_loaded"] = False
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

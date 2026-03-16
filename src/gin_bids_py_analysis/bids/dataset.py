from __future__ import annotations

from pathlib import Path
from typing import Any

from bids import BIDSLayout

from .file import BIDSFile
from .subject import BIDSSubject


class BIDSDataset:
    """
    Wraps a :class:`pybids.BIDSLayout` to provide a typed, queryable BIDS dataset.

    Derivatives are included by default (``derivatives=True``), meaning source
    data and derivative data are queryable through the same instance.

    Args:
        root:           Path to the BIDS dataset root.
        derivatives:    Passed directly to ``BIDSLayout``. ``True`` discovers all
                        ``derivatives/`` sub-datasets; pass a path or list of paths
                        to restrict which derivatives are indexed.
        **layout_kwargs: Additional keyword arguments forwarded to ``BIDSLayout``.

    Example::

        ds = BIDSDataset("/data/my_study")  # includes derivatives by default
        files = ds.get_files(subject="01", suffix="ieeg")
        raw_files = ds.get_files(pipeline="raw", subject="01", suffix="ieeg")
        hilbert_files = ds.get_files(pipeline="hilbert", suffix="timeseries")
        subjects = ds.get_subjects()
        raw_subjects = ds.get_subjects(pipeline="raw")
    """

    def __init__(
        self,
        root: str | Path,
        derivatives: bool | str | list[str] = True,
        **layout_kwargs: Any,
    ) -> None:
        self._root = Path(root)
        self._layout = BIDSLayout(
            str(self._root),
            derivatives=derivatives,
            **layout_kwargs,
        )

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    def get_files(self, pipeline: str | None = None, **entities: Any) -> list[BIDSFile]:
        """
        Return all :class:`BIDSFile` objects matching the given entity filters.

        Any BIDS entity key is valid.

        Args:
            pipeline:   Restrict results to a specific data source. Pass ``"raw"``
                        for source data only, a derivative pipeline name (e.g.
                        ``"hilbert"``) for a specific derivative, or ``None``
                        (default) to query across all sources.
            **entities: BIDS entity filters (e.g. ``subject="01"``,
                        ``suffix="ieeg"``).

        Example::

            dataset.get_files(subject="01", suffix="ieeg", extension=".vhdr")
            dataset.get_files(pipeline="raw", subject="01", suffix="ieeg")
            dataset.get_files(pipeline="hilbert", suffix="timeseries")
        """
        scope_kwargs = {"scope": pipeline} if pipeline is not None else {}
        raw = self._layout.get(**entities, return_type="object", **scope_kwargs)
        return [BIDSFile(f) for f in raw]

    def get_subjects(self, pipeline: str | None = None) -> list[BIDSSubject]:
        """
        Return one :class:`BIDSSubject` per participant found in the dataset.

        Args:
            pipeline:   Restrict to subjects that have files in the given
                        pipeline. See :meth:`get_files` for accepted values.
        """
        scope_kwargs = {"scope": pipeline} if pipeline is not None else {}
        subject_ids = self._layout.get_subjects(**scope_kwargs)
        return [
            BIDSSubject(subject_id=sid, files=self.get_files(subject=sid, pipeline=pipeline))
            for sid in subject_ids
        ]

    def get_subject(self, subject_id: str, pipeline: str | None = None) -> BIDSSubject:
        """
        Return a single :class:`BIDSSubject` by participant ID.

        Args:
            subject_id: Participant label (without ``sub-`` prefix).
            pipeline:   Restrict the subject's files to the given pipeline.
                        See :meth:`get_files` for accepted values.
        """
        return BIDSSubject(subject_id=subject_id, files=self.get_files(subject=subject_id, pipeline=pipeline))

    # ------------------------------------------------------------------
    # Passthrough
    # ------------------------------------------------------------------

    @property
    def layout(self) -> BIDSLayout:
        """Direct access to the underlying ``pybids.BIDSLayout`` for advanced use."""
        return self._layout

    def __repr__(self) -> str:
        return f"BIDSDataset(root={str(self._root)!r})"

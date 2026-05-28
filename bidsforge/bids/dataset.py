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
        raw_files = ds.get_files(scope="raw", subject="01", suffix="ieeg")
        hilbert_files = ds.get_files(scope="hilbert", suffix="timeseries")
        subjects = ds.get_subjects()
        raw_subjects = ds.get_subjects(scope="raw")
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

    def get_files(self, **entities: Any) -> list[BIDSFile]:
        """
        Return all :class:`BIDSFile` objects matching the given entity filters.

        Any BIDS entity key accepted by :meth:`pybids.BIDSLayout.get` is valid,
        including ``scope``.

        Args:
            **entities: BIDS entity filters (e.g. ``subject="01"``,
                        ``suffix="ieeg"``, ``scope="hilbert"``).

        Example::

            dataset.get_files(subject="01", suffix="ieeg", extension=".vhdr")
            dataset.get_files(scope="raw", subject="01", suffix="ieeg")
            dataset.get_files(scope="hilbert", suffix="timeseries")
        """
        raw = self._layout.get(return_type="object", invalid_filters="allow", **entities)
        return [BIDSFile(f) for f in raw]

    def get_subjects(self, **entities: Any) -> list[BIDSSubject]:
        """
        Return one :class:`BIDSSubject` per participant found in the dataset.

        Args:
            **entities: Filters forwarded to the subject lookup. This accepts the
                        same query keys as :meth:`get_files`, including
                        ``scope``.
        """
        subject_ids = self._layout.get(
            return_type="id", target="subject", invalid_filters="allow", **entities
        )
        return [
            BIDSSubject(
                subject_id=sid,
                files=self.get_files(**{**entities, "subject": sid}),
            )
            for sid in subject_ids
        ]

    def get_subject(self, subject_id: str, **entities: Any) -> BIDSSubject:
        """
        Return a single :class:`BIDSSubject` by participant ID.

        Args:
            subject_id: Participant label (without ``sub-`` prefix).
            **entities: Filters forwarded to :meth:`get_files`, including
                        ``scope``.
        """
        return BIDSSubject(
            subject_id=subject_id,
            files=self.get_files(**{**entities, "subject": subject_id}),
        )

    # ------------------------------------------------------------------
    # Passthrough
    # ------------------------------------------------------------------

    @property
    def layout(self) -> BIDSLayout:
        """Direct access to the underlying ``pybids.BIDSLayout`` for advanced use."""
        return self._layout

    def __repr__(self) -> str:
        return f"BIDSDataset(root={str(self._root)!r})"

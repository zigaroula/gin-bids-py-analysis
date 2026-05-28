"""Bridge between in-memory ConditionTestProcessingResult and ConditionTestGroupProcessing.

``ConditionTestGroupProcessing.process_group()`` reads HDF5 files through the
``BIDSFile.ensure_loaded()`` interface.  This module constructs a fully
in-memory ``h5py.File`` (``driver="core", backing_store=False``) that matches
the schema read by ``_load_raw_from_hdf5``, attaches it to a synthetic
``BIDSFile``, and builds the ``BIDSFileGroup`` expected by the processor
— all without any disk I/O.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Generator
from uuid import uuid4

import h5py
import numpy as np

from bidsforge.bids.file import BIDSFile
from bidsforge.processing.trial_stats_group import (
    build_condition_test_compatible_groups,
)

if TYPE_CHECKING:
    from bidsforge.bids.file_group import BIDSFileGroup
    from bidsforge.processing.trial_stats import (
        ConditionTestProcessingResult,
    )


# ---------------------------------------------------------------------------
# Private mock pybids file
# ---------------------------------------------------------------------------


class _MockPyBIDSFile:
    """Minimal stand-in for a pybids BIDSFile — no pybids/SQLAlchemy required."""

    def __init__(self, path: str, entities: dict) -> None:
        self.path = path
        self.entities = entities


# ---------------------------------------------------------------------------
# HDF5 structure writer
# ---------------------------------------------------------------------------


def _write_hdf5_structure(
    fh: h5py.File,
    result: "ConditionTestProcessingResult",
    subject_id: str,
) -> None:
    """Populate *fh* with the datasets read by ``_load_raw_from_hdf5``.

    The written schema matches exactly what
    ``ConditionTestGroupProcessing._load_raw_from_hdf5`` reads:

    * ``axes/channel``                          — channel name array
    * ``axes/time_s``                           — time axis
    * ``meta/analysis_level``                   — "channel"
    * ``meta/condition_labels``                 — [condition_a, condition_b]
    * ``meta/binning_mode``                     — "window_ms" / "n_bins" / "none"
    * ``meta/window_ms``                        — float
    * ``meta/n_bins``                           — int
    * ``meta/effective_n_bins``                 — int (= n_times)
    * ``stats/condition_contrast/t_values``     — (n_channels, n_times) float64
    * ``data/signal_activity/difference/mean``         — (n_channels, n_times) float64
    * ``data/signal_activity/condition_a/mean``        — (n_channels, n_times) float64
    * ``data/signal_activity/condition_b/mean``        — (n_channels, n_times) float64
    * ``provenance/source_ieeg_files``          — string array
    * ``provenance/source_electrodes_files``    — string array
    """
    str_dt = h5py.string_dtype(encoding="utf-8")

    # axes
    axes = fh.create_group("axes")
    axes.create_dataset(
        "channel",
        data=np.array(result.channel_names, dtype=object),
        dtype=str_dt,
    )
    axes.create_dataset("time_s", data=result.time_axis_s.astype(np.float64))

    # meta
    meta = fh.create_group("meta")
    meta.create_dataset("schema_version", data=np.bytes_("3.0"))
    meta.create_dataset("analysis_level", data=np.bytes_("channel"))
    meta.create_dataset(
        "condition_labels",
        data=np.array([result.condition_a, result.condition_b], dtype=object),
        dtype=str_dt,
    )
    binning_mode = (
        "window_ms"
        if result.window_ms > 0
        else ("n_bins" if result.n_bins > 0 else "none")
    )
    meta.create_dataset("binning_mode", data=np.bytes_(binning_mode))
    meta.create_dataset("window_ms", data=float(result.window_ms))
    meta.create_dataset("n_bins", data=int(result.n_bins))
    n_times = result.time_axis_s.shape[0]
    meta.create_dataset("effective_n_bins", data=n_times)
    meta.create_dataset("activity_zscore", data=np.bytes_(str(result.activity_zscore)))
    meta.create_dataset(
        "activity_baseline_tmin_s",
        data=float(result.activity_baseline_tmin_s),
    )
    meta.create_dataset(
        "activity_baseline_tmax_s",
        data=float(result.activity_baseline_tmax_s),
    )

    # stats
    stats_grp = fh.create_group("stats")
    contrast_grp = stats_grp.create_group("condition_contrast")
    t = result.contrast.t_values
    if t.size == 0:
        t = np.zeros((len(result.channel_names), n_times), dtype=np.float64)
    contrast_grp.create_dataset("t_values", data=t.astype(np.float64))
    if result.contrast.permuted_t_values is not None:
        contrast_grp.create_dataset(
            "permuted_t_values",
            data=result.contrast.permuted_t_values.astype(np.float32),
        )

    # data/signal_activity
    data_grp = fh.create_group("data")
    activity_grp = data_grp.create_group("signal_activity")
    diff = result.difference.mean
    if diff.size == 0:
        diff = np.zeros_like(t)
    diff_grp = activity_grp.create_group("difference")
    diff_grp.create_dataset("mean", data=diff.astype(np.float64))
    cond_a = result.signal_activity.condition_a.mean
    cond_b = result.signal_activity.condition_b.mean
    if cond_a.size == 0:
        cond_a = np.zeros_like(t)
    if cond_b.size == 0:
        cond_b = np.zeros_like(t)
    cond_a_grp = activity_grp.create_group("condition_a")
    cond_a_grp.create_dataset("mean", data=cond_a.astype(np.float64))
    cond_b_grp = activity_grp.create_group("condition_b")
    cond_b_grp.create_dataset("mean", data=cond_b.astype(np.float64))

    # provenance
    prov = fh.create_group("provenance")
    prov.create_dataset(
        "source_ieeg_files",
        data=np.array(result.source_ieeg_files or [], dtype=object),
        dtype=str_dt,
    )
    prov.create_dataset(
        "source_electrodes_files",
        data=np.array(result.source_electrodes_files or [], dtype=object),
        dtype=str_dt,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_in_memory_bids_file_with_handle(
    result: "ConditionTestProcessingResult",
    subject_id: str,
) -> tuple[BIDSFile, h5py.File]:
    """Create an in-memory BIDSFile and return it together with the h5py handle.

    The caller is responsible for closing the returned ``h5py.File`` when the
    file is no longer needed.  Prefer :func:`group_file_group_context` which
    manages the lifetime automatically.
    """
    fake_path = Path(f"/in-memory/sub-{subject_id}_{uuid4().hex}_stats.h5")
    mock = _MockPyBIDSFile(
        path=str(fake_path),
        entities={
            "subject": subject_id,
            "suffix": "stats",
            "extension": ".h5",
        },
    )
    bids_file = BIDSFile(mock)
    fh: h5py.File = h5py.File(str(fake_path), "w", driver="core", backing_store=False)
    _write_hdf5_structure(fh, result, subject_id)
    bids_file.attach_data(fh)
    return bids_file, fh


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_in_memory_bids_file(
    result: "ConditionTestProcessingResult",
    subject_id: str,
) -> BIDSFile:
    """Wrap one ``ConditionTestProcessingResult`` as a ``BIDSFile`` with an
    in-memory HDF5 file attached.

    The returned ``BIDSFile`` behaves exactly like one pointing at a real
    ``*_stats.h5`` derivative: ``file.extension`` is ``".h5"``,
    ``file.get("subject")`` returns *subject_id*, and ``file.ensure_loaded()``
    yields the in-memory ``h5py.File`` handle.

    No disk I/O is performed.

    .. note::
        The caller owns the h5py handle embedded in the returned file.  For
        production use where the file group has a bounded lifetime, prefer
        :func:`group_file_group_context` which closes every handle automatically.
    """
    bids_file, _ = _build_in_memory_bids_file_with_handle(result, subject_id)
    return bids_file


@contextmanager
def group_file_group_context(
    results: dict[str, "ConditionTestProcessingResult"],
    primary_condition_metric: str,
) -> Generator["BIDSFileGroup", None, None]:
    """Context manager that builds a ``BIDSFileGroup`` from in-memory results
    and guarantees that every h5py handle is closed on exit.

    Use this instead of :func:`build_group_file_group_from_results` whenever
    the group is only needed for a single :meth:`process_group` call so that
    h5py core-driver memory is reclaimed immediately.

    Parameters
    ----------
    results:
        In-memory subject results produced by ``ConditionTestProcessing``.
    primary_condition_metric:
        The metric that ``ConditionTestGroupProcessing`` will read.

    Yields
    ------
    BIDSFileGroup
        A single group spanning all subjects.

    Raises
    ------
    ValueError
        If *results* is empty or incompatible.
    """
    if not results:
        raise ValueError("group_file_group_context requires at least one result.")

    handles: list[h5py.File] = []
    try:
        bids_files: list[BIDSFile] = []
        for subject_id, result in sorted(results.items()):
            bids_file, fh = _build_in_memory_bids_file_with_handle(result, subject_id)
            handles.append(fh)
            bids_files.append(bids_file)

        groups = build_condition_test_compatible_groups(
            bids_files,
            primary_condition_metric=primary_condition_metric,
        )
        if not groups:
            raise ValueError(
                "build_condition_test_compatible_groups returned no groups for the provided results."
            )
        yield max(groups, key=lambda g: len(g.all_files))
    finally:
        for fh in handles:
            try:
                if fh.id.valid:
                    fh.close()
            except Exception:  # noqa: BLE001
                pass


def build_group_file_group_from_results(
    results: dict[str, "ConditionTestProcessingResult"],
    primary_condition_metric: str,
) -> "BIDSFileGroup":
    """Build a ``BIDSFileGroup`` from a mapping of *subject_id → result*.

    .. warning::
        The h5py handles embedded in the returned group are **not** closed by
        this function.  This is intentional for test use where the group
        outlives the function call.  For the interactive app use
        :func:`group_file_group_context` instead.

    Parameters
    ----------
    results:
        In-memory subject results produced by `ConditionTestProcessing`.
    primary_condition_metric:
        The metric that ``ConditionTestGroupProcessing`` will read (e.g.
        ``"t_values"``, ``"mean_difference"``).

    Returns
    -------
    BIDSFileGroup
        A single group spanning all subjects.

    Raises
    ------
    ValueError
        If *results* is empty, or if the results are mutually incompatible.
    """
    if not results:
        raise ValueError("build_group_file_group_from_results requires at least one result.")

    bids_files = [
        build_in_memory_bids_file(result, subject_id)
        for subject_id, result in sorted(results.items())
    ]

    groups = build_condition_test_compatible_groups(
        bids_files,
        primary_condition_metric=primary_condition_metric,
    )
    if not groups:
        raise ValueError(
            "build_condition_test_compatible_groups returned no groups for the provided results."
        )
    return max(groups, key=lambda g: len(g.all_files))



from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import mne


def load_ieeg(path: Path | str, preload: bool = True, **kwargs: Any) -> mne.io.BaseRaw:
    """Load iEEG data from a file path.

    Uses :func:`mne.io.read_raw` for automatic format detection based on the
    file extension.  Supported formats include BrainVision (``.vhdr``), EDF
    (``.edf``), BDF (``.bdf``), FIFF (``.fif``), EEGLAB (``.set``), and any
    other format recognised by MNE ≥ 1.0.

    Args:
        path:     Path to the raw iEEG recording file.
        preload:  If ``True`` (default) the data are read into memory immediately.
                  Pass ``False`` to keep data memory-mapped on disk — useful for
                  very large files.
        **kwargs: Forwarded verbatim to :func:`mne.io.read_raw` (e.g.
                  ``verbose=False``, ``allow_maxshield=True``).

    Returns:
        An :class:`mne.io.BaseRaw` subclass instance (the concrete type depends
        on the file format detected by MNE).

    Note:
        Inside processors, prefer
        :meth:`BIDSFile.ensure_loaded() <gin_bids_py_analysis.bids.file.BIDSFile.ensure_loaded>`
        which reuses already-attached data when available and discards
        on-demand loaded data automatically after the context exits.

    Example::

        from pathlib import Path
        from gin_bids_py_analysis.data import load_ieeg

        raw = load_ieeg(Path("/data/sub-01/ieeg/sub-01_task-rest_ieeg.vhdr"))
        print(raw.info)
    """
    return mne.io.read_raw(str(path), preload=preload, **kwargs)


def load_table(path: Path | str, **kwargs: Any) -> list[dict[str, str]]:
    """Load a TSV or CSV table file into a list of row dicts.

    Delimiter is inferred from the file extension: ``.tsv`` → tab, all others
    → comma.  Each value is stripped of surrounding whitespace; ``None`` (a
    missing field in :mod:`csv`) is coerced to ``""``.

    Args:
        path:     Path to the ``.tsv`` or ``.csv`` file.
        **kwargs: Reserved for future options; currently unused.

    Returns:
        A list where each element is a ``dict[str, str]`` mapping column
        header to cell value for one row.

    Example::

        from pathlib import Path
        from gin_bids_py_analysis.data import load_table

        rows = load_table(Path("/data/sub-01/ieeg/sub-01_electrodes.tsv"))
        for row in rows:
            print(row["name"], row.get("region"))
    """
    path = Path(path)
    delimiter = "\t" if path.suffix == ".tsv" else ","
    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        return [
            {
                str(key).strip(): "" if value is None else str(value).strip()
                for key, value in row.items()
            }
            for row in reader
        ]


def load_json(path: Path | str, **kwargs: Any) -> dict:
    """Load a JSON file and return its top-level object as a :class:`dict`.

    Args:
        path:     Path to the ``.json`` file.
        **kwargs: Reserved for future options; currently unused.

    Returns:
        The parsed JSON object (always a :class:`dict` for well-formed BIDS
        sidecar files).

    Example::

        from pathlib import Path
        from gin_bids_py_analysis.data import load_json

        meta = load_json(Path("/data/sub-01/ieeg/sub-01_task-rest_ieeg.json"))
        print(meta.get("SamplingFrequency"))
    """
    with open(Path(path), "r", encoding="utf-8") as fh:
        return json.load(fh)

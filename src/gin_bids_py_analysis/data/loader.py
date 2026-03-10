from __future__ import annotations

from typing import Any

import mne

from gin_bids_py_analysis.bids.file import BIDSFile


def load_ieeg(bids_file: BIDSFile, preload: bool = True, **kwargs: Any) -> mne.io.BaseRaw:
    """Load iEEG data from a :class:`~gin_bids_py_analysis.bids.file.BIDSFile`.

    Uses :func:`mne.io.read_raw` for automatic format detection based on the
    file extension.  Supported formats include BrainVision (``.vhdr``), EDF
    (``.edf``), BDF (``.bdf``), FIFF (``.fif``), EEGLAB (``.set``), and any
    other format recognised by MNE ≥ 1.0.

    Args:
        bids_file: A :class:`BIDSFile` whose ``.path`` points to the raw recording.
        preload:   If ``True`` (default) the data are read into memory immediately.
                   Pass ``False`` to keep data memory-mapped on disk — useful for
                   very large files.
        **kwargs:  Forwarded verbatim to :func:`mne.io.read_raw` (e.g.
                   ``verbose=False``, ``allow_maxshield=True``).

    Returns:
        An :class:`mne.io.BaseRaw` subclass instance (the concrete type depends
        on the file format detected by MNE).

    Example::

        from gin_bids_py_analysis.data import load_ieeg

        raw = load_ieeg(bids_file)
        print(raw.info)
    """
    return mne.io.read_raw(str(bids_file.path), preload=preload, **kwargs)

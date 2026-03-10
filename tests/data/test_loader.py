"""Unit tests for gin_bids_py_analysis.data.loader.

All tests are mock-based: no real file or MNE installation is required
to run these (mne is patched at import time).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gin_bids_py_analysis.data import load_ieeg


def _make_bids_file(path: str = "/data/sub-01_ieeg.vhdr") -> MagicMock:
    """Return a minimal BIDSFile mock with a .path property."""
    bids_file = MagicMock()
    bids_file.path = Path(path)
    return bids_file


class TestLoadIeeg:
    def test_calls_mne_read_raw_with_path_string(self):
        bids_file = _make_bids_file("/data/sub-01_ieeg.vhdr")
        with patch("gin_bids_py_analysis.data.loader.mne.io.read_raw") as mock_read:
            load_ieeg(bids_file)
        mock_read.assert_called_once()
        call_args = mock_read.call_args
        assert call_args.args[0] == str(bids_file.path)

    def test_preload_true_by_default(self):
        bids_file = _make_bids_file()
        with patch("gin_bids_py_analysis.data.loader.mne.io.read_raw") as mock_read:
            load_ieeg(bids_file)
        assert mock_read.call_args.kwargs["preload"] is True

    def test_preload_false_is_forwarded(self):
        bids_file = _make_bids_file()
        with patch("gin_bids_py_analysis.data.loader.mne.io.read_raw") as mock_read:
            load_ieeg(bids_file, preload=False)
        assert mock_read.call_args.kwargs["preload"] is False

    def test_extra_kwargs_forwarded_to_mne(self):
        bids_file = _make_bids_file()
        with patch("gin_bids_py_analysis.data.loader.mne.io.read_raw") as mock_read:
            load_ieeg(bids_file, verbose=False, allow_maxshield=True)
        kwargs = mock_read.call_args.kwargs
        assert kwargs["verbose"] is False
        assert kwargs["allow_maxshield"] is True

    def test_returns_mne_raw_object(self):
        bids_file = _make_bids_file()
        fake_raw = MagicMock()
        with patch(
            "gin_bids_py_analysis.data.loader.mne.io.read_raw", return_value=fake_raw
        ):
            result = load_ieeg(bids_file)
        assert result is fake_raw

    def test_exported_from_data_package(self):
        from gin_bids_py_analysis.data import load_ieeg as _load_ieeg  # noqa: PLC0415
        assert callable(_load_ieeg)

"""Unit tests for gin_bids_py_analysis.data.loader.

All tests are mock-based: no real file or MNE/file-system access is required
to run these (mne and builtins.open are patched as needed).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from gin_bids_py_analysis.data import load_ieeg, load_json, load_table


class TestLoadIeeg:
    def test_calls_mne_read_raw_with_path_string(self):
        path = Path("/data/sub-01_ieeg.vhdr")
        with patch("gin_bids_py_analysis.data.loader.mne.io.read_raw") as mock_read:
            load_ieeg(path)
        mock_read.assert_called_once()
        assert mock_read.call_args.args[0] == str(path)

    def test_accepts_str_path(self):
        with patch("gin_bids_py_analysis.data.loader.mne.io.read_raw") as mock_read:
            load_ieeg("/data/sub-01_ieeg.vhdr")
        assert mock_read.call_args.args[0] == "/data/sub-01_ieeg.vhdr"

    def test_preload_true_by_default(self):
        with patch("gin_bids_py_analysis.data.loader.mne.io.read_raw") as mock_read:
            load_ieeg(Path("/data/sub-01_ieeg.vhdr"))
        assert mock_read.call_args.kwargs["preload"] is True

    def test_preload_false_is_forwarded(self):
        with patch("gin_bids_py_analysis.data.loader.mne.io.read_raw") as mock_read:
            load_ieeg(Path("/data/sub-01_ieeg.vhdr"), preload=False)
        assert mock_read.call_args.kwargs["preload"] is False

    def test_extra_kwargs_forwarded_to_mne(self):
        with patch("gin_bids_py_analysis.data.loader.mne.io.read_raw") as mock_read:
            load_ieeg(Path("/data/sub-01_ieeg.vhdr"), verbose=False, allow_maxshield=True)
        kwargs = mock_read.call_args.kwargs
        assert kwargs["verbose"] is False
        assert kwargs["allow_maxshield"] is True

    def test_returns_mne_raw_object(self):
        fake_raw = MagicMock()
        with patch(
            "gin_bids_py_analysis.data.loader.mne.io.read_raw", return_value=fake_raw
        ):
            result = load_ieeg(Path("/data/sub-01_ieeg.vhdr"))
        assert result is fake_raw

    def test_exported_from_data_package(self):
        from gin_bids_py_analysis.data import load_ieeg as _load_ieeg  # noqa: PLC0415

        assert callable(_load_ieeg)


class TestLoadTable:
    def _tsv_content(self) -> str:
        return "name\tregion\nA1\tFrontal\nA2\tTemporal\n"

    def _csv_content(self) -> str:
        return "name,region\nA1,Frontal\nA2,Temporal\n"

    def test_tsv_uses_tab_delimiter(self):
        content = self._tsv_content()
        m = mock_open(read_data=content)
        with patch("builtins.open", m):
            rows = load_table(Path("/data/electrodes.tsv"))
        assert rows == [
            {"name": "A1", "region": "Frontal"},
            {"name": "A2", "region": "Temporal"},
        ]

    def test_csv_uses_comma_delimiter(self):
        content = self._csv_content()
        m = mock_open(read_data=content)
        with patch("builtins.open", m):
            rows = load_table(Path("/data/electrodes.csv"))
        assert rows == [
            {"name": "A1", "region": "Frontal"},
            {"name": "A2", "region": "Temporal"},
        ]

    def test_non_tsv_extension_defaults_to_comma(self):
        content = "a,b\n1,2\n"
        m = mock_open(read_data=content)
        with patch("builtins.open", m):
            rows = load_table(Path("/data/file.txt"))
        assert rows == [{"a": "1", "b": "2"}]

    def test_accepts_str_path(self):
        m = mock_open(read_data="a\t b\n1\t2\n")
        with patch("builtins.open", m):
            rows = load_table("/data/file.tsv")
        assert rows[0] == {"a": "1", "b": "2"}

    def test_values_are_stripped(self):
        m = mock_open(read_data="col\n  hello  \n")
        with patch("builtins.open", m):
            rows = load_table(Path("/data/file.csv"))
        assert rows[0]["col"] == "hello"

    def test_exported_from_data_package(self):
        from gin_bids_py_analysis.data import load_table as _load_table  # noqa: PLC0415

        assert callable(_load_table)


class TestLoadJson:
    def test_returns_dict(self):
        data = {"SamplingFrequency": 1000, "Channels": ["A1", "A2"]}
        content = json.dumps(data)
        m = mock_open(read_data=content)
        with patch("builtins.open", m):
            result = load_json(Path("/data/sidecar.json"))
        assert result == data

    def test_accepts_str_path(self):
        m = mock_open(read_data='{"key": "value"}')
        with patch("builtins.open", m):
            result = load_json("/data/sidecar.json")
        assert result == {"key": "value"}

    def test_exported_from_data_package(self):
        from gin_bids_py_analysis.data import load_json as _load_json  # noqa: PLC0415

        assert callable(_load_json)

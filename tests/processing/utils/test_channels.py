"""Tests for select_channels_for_montage in processing.utils.channels."""

from __future__ import annotations

import re

import numpy as np
import pytest

from gin_bids_py_analysis.processing.utils.channels import select_channels_for_montage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _data(n_ch: int = 4, n_samples: int = 8) -> np.ndarray:
    return np.arange(n_ch * n_samples, dtype=float).reshape(n_ch, n_samples)


NAMES = ["A1", "A2", "Bp1", "Bp2", "X10", "X20"]
DATA = _data(len(NAMES))


# ---------------------------------------------------------------------------
# None — no filtering
# ---------------------------------------------------------------------------

class TestNoFilter:
    def test_none_returns_all_channels(self) -> None:
        out_data, out_names = select_channels_for_montage(DATA, NAMES, None)
        np.testing.assert_array_equal(out_data, DATA)
        assert out_names == NAMES

    def test_empty_list_returns_all_channels(self) -> None:
        out_data, out_names = select_channels_for_montage(DATA, NAMES, [])
        np.testing.assert_array_equal(out_data, DATA)
        assert out_names == NAMES


# ---------------------------------------------------------------------------
# list[str] — exact-name allowlist (original behaviour)
# ---------------------------------------------------------------------------

class TestExactList:
    def test_selects_named_channels(self) -> None:
        out_data, out_names = select_channels_for_montage(DATA, NAMES, ["A1", "X10"])
        assert out_names == ["A1", "X10"]
        np.testing.assert_array_equal(out_data, DATA[[0, 4], :])

    def test_preserves_file_order_not_list_order(self) -> None:
        # Requested in reversed order — output should still be file order
        out_data, out_names = select_channels_for_montage(DATA, NAMES, ["X10", "A1"])
        assert out_names == ["A1", "X10"]

    def test_raises_when_no_match(self) -> None:
        with pytest.raises(ValueError, match="channels_for_montage did not match any input channels"):
            select_channels_for_montage(DATA, NAMES, ["Z99"])


# ---------------------------------------------------------------------------
# str — regex pattern (new behaviour)
# ---------------------------------------------------------------------------

class TestRegexStr:
    def test_simple_prefix_pattern(self) -> None:
        # Match channels starting with 'A'
        out_data, out_names = select_channels_for_montage(DATA, NAMES, r"A\d+")
        assert out_names == ["A1", "A2"]
        np.testing.assert_array_equal(out_data, DATA[[0, 1], :])

    def test_optional_letter_pattern(self) -> None:
        # Match 'Bp1' and 'Bp2' but not 'A1', 'A2', 'X10', 'X20'
        out_data, out_names = select_channels_for_montage(DATA, NAMES, r"[A-Za-z]p\d+")
        assert out_names == ["Bp1", "Bp2"]

    def test_index_below_20(self) -> None:
        # Simulated user example: one letter + optional 'p' + integer 1-19
        names = ["A1", "A10", "A19", "A20", "Bp5", "Bp20", "X1"]
        data = _data(len(names))
        pattern = r"[A-Za-z]p?([1-9]|1[0-9])"
        out_data, out_names = select_channels_for_montage(data, names, pattern)
        assert out_names == ["A1", "A10", "A19", "Bp5", "X1"]

    def test_preserves_file_order(self) -> None:
        # Pattern matches A2 and A1 — output must follow file order
        out_data, out_names = select_channels_for_montage(DATA, NAMES, r"A[12]")
        assert out_names == ["A1", "A2"]

    def test_raises_when_pattern_matches_nothing(self) -> None:
        with pytest.raises(ValueError, match="channels_for_montage pattern did not match any input channels"):
            select_channels_for_montage(DATA, NAMES, r"Z\d+")

    def test_fullmatch_does_not_match_partial(self) -> None:
        # 'A' alone must NOT match 'A1' (fullmatch, not search)
        with pytest.raises(ValueError, match="pattern did not match"):
            select_channels_for_montage(DATA, NAMES, r"A")


# ---------------------------------------------------------------------------
# re.Pattern — compiled pattern (new behaviour)
# ---------------------------------------------------------------------------

class TestCompiledPattern:
    def test_compiled_pattern_selects_channels(self) -> None:
        pattern = re.compile(r"X\d+", re.IGNORECASE)
        out_data, out_names = select_channels_for_montage(DATA, NAMES, pattern)
        assert out_names == ["X10", "X20"]

    def test_compiled_pattern_raises_when_no_match(self) -> None:
        pattern = re.compile(r"Z\d+")
        with pytest.raises(ValueError, match="pattern did not match"):
            select_channels_for_montage(DATA, NAMES, pattern)

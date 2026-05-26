from __future__ import annotations

import numpy as np
import pytest

from gin_bids_py_analysis.processing.utils.matlab import (
    mat_float,
    mat_int,
    mat_str,
    mat_str_list,
    matlab_round,
)


class TestMatStr:
    def test_plain_str(self) -> None:
        assert mat_str("hello") == "hello"

    def test_bytes(self) -> None:
        assert mat_str(b"hello") == "hello"

    def test_zero_d_array_str(self) -> None:
        arr = np.array("hello", dtype=object)
        assert mat_str(arr) == "hello"

    def test_zero_d_array_bytes(self) -> None:
        arr = np.array(b"hello", dtype=object)
        assert mat_str(arr) == "hello"

    def test_one_d_single_element_array(self) -> None:
        # squeeze_me=True collapses (1,) arrays to scalars for most types but
        # object arrays may stay as (1,) — handle both.
        arr = np.array(["hello"], dtype=object)
        assert mat_str(arr) == "hello"

    def test_empty_array_returns_default(self) -> None:
        arr = np.array([], dtype=object)
        assert mat_str(arr, default="fallback") == "fallback"

    def test_none_returns_default(self) -> None:
        assert mat_str(None, default="x") == "x"


class TestMatlabRound:
    def test_half_values_round_away_from_zero(self) -> None:
        assert matlab_round(1.5) == 2
        assert matlab_round(2.5) == 3
        assert matlab_round(-1.5) == -2
        assert matlab_round(-2.5) == -3

    def test_non_half_values_round_to_nearest(self) -> None:
        assert matlab_round(1.49) == 1
        assert matlab_round(1.51) == 2
        assert matlab_round(-1.49) == -1
        assert matlab_round(-1.51) == -2


class TestMatFloat:
    def test_plain_float(self) -> None:
        assert mat_float(1.5, default=0.0) == pytest.approx(1.5)

    def test_zero_d_array(self) -> None:
        arr = np.array(3.14)
        assert mat_float(arr, default=0.0) == pytest.approx(3.14)

    def test_one_d_array(self) -> None:
        arr = np.array([2.0])
        assert mat_float(arr, default=0.0) == pytest.approx(2.0)

    def test_empty_array_returns_default(self) -> None:
        arr = np.array([])
        assert mat_float(arr, default=-1.0) == pytest.approx(-1.0)

    def test_none_returns_default(self) -> None:
        assert mat_float(None, default=99.0) == pytest.approx(99.0)


class TestMatInt:
    def test_plain_int(self) -> None:
        assert mat_int(7, default=0) == 7

    def test_zero_d_array(self) -> None:
        arr = np.array(42)
        assert mat_int(arr, default=0) == 42

    def test_empty_array_returns_default(self) -> None:
        arr = np.array([])
        assert mat_int(arr, default=-1) == -1

    def test_none_returns_default(self) -> None:
        assert mat_int(None, default=5) == 5


class TestMatStrList:
    def test_plain_list_from_ndarray(self) -> None:
        arr = np.array(["A1", "B2", "C3"], dtype=object)
        assert mat_str_list(arr) == ["A1", "B2", "C3"]

    def test_bytes_array(self) -> None:
        arr = np.array([b"A1", b"B2"], dtype=object)
        assert mat_str_list(arr) == ["A1", "B2"]

    def test_bare_string_scalar(self) -> None:
        # squeeze_me=True with a single-string array yields a bare str
        assert mat_str_list("hello") == ["hello"]

    def test_bare_bytes_scalar(self) -> None:
        assert mat_str_list(b"hello") == ["hello"]

    def test_empty_array(self) -> None:
        arr = np.array([], dtype=object)
        assert mat_str_list(arr) == []

    def test_none_returns_empty(self) -> None:
        assert mat_str_list(None) == []

    def test_two_d_array_flattened(self) -> None:
        # savemat may produce 2-D object arrays for string lists in some cases
        arr = np.array([["A1"], ["B2"]], dtype=object)
        assert mat_str_list(arr) == ["A1", "B2"]




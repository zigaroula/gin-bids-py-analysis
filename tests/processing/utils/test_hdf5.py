from __future__ import annotations

import h5py
import numpy as np
import pytest

from gin_bids_py_analysis.processing.utils.hdf5 import (
    coerce_feature_time,
    dataset_or_none,
    decode_str_array,
    float_scalar,
    int_scalar,
    str_scalar,
)


def test_dataset_or_none_returns_nested_dataset() -> None:
    with h5py.File("inmem.h5", "w", driver="core", backing_store=False) as fh:
        grp = fh.create_group("meta")
        grp.create_dataset("n_bins", data=12)
        found = dataset_or_none(fh, "meta/n_bins")
        missing = dataset_or_none(fh, "meta/unknown")

        assert found is not None
        assert int(found[()]) == 12
        assert missing is None


def test_decode_str_array_supports_bytes_and_str() -> None:
    values = np.array([b"A1", "B2"], dtype=object)
    assert decode_str_array(values) == ["A1", "B2"]


def test_scalar_helpers_fallback_to_default() -> None:
    with h5py.File("inmem.h5", "w", driver="core", backing_store=False) as fh:
        fh.create_dataset("int_ok", data=7)
        fh.create_dataset("float_ok", data=2.5)
        fh.create_dataset("str_ok", data="hello", dtype=h5py.string_dtype(encoding="utf-8"))
        fh.create_dataset("not_int", data="abc", dtype=h5py.string_dtype(encoding="utf-8"))

        assert int_scalar(dataset_or_none(fh, "int_ok"), default=0) == 7
        assert float_scalar(dataset_or_none(fh, "float_ok"), default=0.0) == 2.5
        assert str_scalar(dataset_or_none(fh, "str_ok"), default="") == "hello"
        assert int_scalar(dataset_or_none(fh, "not_int"), default=11) == 11
        assert float_scalar(dataset_or_none(fh, "missing"), default=3.0) == 3.0
        assert str_scalar(dataset_or_none(fh, "missing"), default="na") == "na"


def test_coerce_feature_time_handles_transposed_and_flat_inputs() -> None:
    transposed = np.array(
        [
            [1.0, 4.0],
            [2.0, 5.0],
            [3.0, 6.0],
        ]
    )
    reshaped = coerce_feature_time(transposed, n_features=2, n_times=3)
    assert reshaped.shape == (2, 3)
    assert np.allclose(reshaped, np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))

    flat = np.array([1.0, 2.0, 3.0, 4.0])
    flat_out = coerce_feature_time(flat, n_features=2, n_times=2)
    assert flat_out.shape == (2, 2)
    assert np.allclose(flat_out, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_coerce_feature_time_rejects_non_1d_2d() -> None:
    with pytest.raises(ValueError, match="unsupported shape"):
        coerce_feature_time(np.zeros((2, 2, 2), dtype=np.float64), n_features=2, n_times=2)

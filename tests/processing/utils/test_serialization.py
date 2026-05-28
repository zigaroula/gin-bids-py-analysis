from __future__ import annotations

import h5py
import numpy as np
import scipy.io

from gin_bids_py_analysis.processing.utils.serialization import (
    compressed,
    write_hdf5_tree,
    write_matlab_tree,
)


def test_write_hdf5_tree_handles_nested_scalars_arrays_and_strings(tmp_path) -> None:
    path = tmp_path / "tree.h5"
    write_hdf5_tree(
        path,
        {
            "meta": {"schema_version": "2.0", "ok": True},
            "axes": {"channel": np.array(["A1", "B2"], dtype=object)},
            "data": {"values": compressed(np.arange(8, dtype=np.float64).reshape(2, 4))},
        },
    )

    with h5py.File(path, "r") as fh:
        assert fh["meta/schema_version"].asstr()[()] == "2.0"
        assert bool(fh["meta/ok"][()]) is True
        assert list(fh["axes/channel"].asstr()[:]) == ["A1", "B2"]
        np.testing.assert_allclose(fh["data/values"][:], np.arange(8).reshape(2, 4))


def test_write_matlab_tree_preserves_nested_structure(tmp_path) -> None:
    path = tmp_path / "tree.mat"
    write_matlab_tree(
        path,
        {
            "meta": {"schema_version": "2.0"},
            "data": {
                "condition_a": {"mean": np.ones((2, 3), dtype=np.float64)},
                "labels": ["A1", "B2"],
            },
        },
    )

    mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    data = mat["data"]
    assert str(data.meta.schema_version) == "2.0"
    assert data.data.condition_a.mean.shape == (2, 3)
    assert list(np.atleast_1d(data.data.labels)) == ["A1", "B2"]


def test_write_matlab_tree_writes_1d_arrays_as_columns(tmp_path) -> None:
    path = tmp_path / "tree.mat"
    write_matlab_tree(
        path,
        {
            "data": {
                "values": np.array([1, 2, 3], dtype=np.int64),
            },
        },
    )

    mat = scipy.io.loadmat(str(path), squeeze_me=False)
    values = mat["data"]["data"][0, 0]["values"][0, 0]
    assert values.shape == (3, 1)




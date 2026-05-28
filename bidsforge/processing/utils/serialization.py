from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, TypeAlias

import h5py
import numpy as np
from numpy.typing import NDArray
from scipy.io import savemat

from bidsforge.processing.utils.matlab import make_struct, matlab_safe_name

if TYPE_CHECKING:
    OutputScalar: TypeAlias = str | bytes | bool | int | float | np.generic[Any]
    OutputValue: TypeAlias = (
        OutputScalar
        | NDArray[Any]
        | list[Any]
        | tuple[Any, ...]
        | Mapping[str, Any]
        | "OutputNode"
        | None
    )
    OutputTree: TypeAlias = Mapping[str, OutputValue]
else:
    OutputScalar = str | bytes | bool | int | float | np.generic
    OutputValue = Any
    OutputTree = Mapping[str, Any]


@dataclass(frozen=True)
class OutputNode:
    """Value wrapper carrying format hints for generic tree serializers."""

    value: OutputValue
    dtype: Any | None = None
    compression: str | None = None
    compression_opts: int | None = None
    attrs: Mapping[str, Any] = field(default_factory=dict)
    matlab_name: str | None = None
    skip_if_none: bool = True


def dataset(
    value: OutputValue,
    *,
    dtype: Any | None = None,
    compression: str | None = None,
    compression_opts: int | None = None,
    attrs: Mapping[str, Any] | None = None,
    matlab_name: str | None = None,
    skip_if_none: bool = True,
) -> OutputNode:
    """Return an output value with optional serialization hints."""

    return OutputNode(
        value=value,
        dtype=dtype,
        compression=compression,
        compression_opts=compression_opts,
        attrs={} if attrs is None else dict(attrs),
        matlab_name=matlab_name,
        skip_if_none=skip_if_none,
    )


def compressed(value: OutputValue, *, level: int = 4) -> OutputNode:
    return dataset(value, compression="gzip", compression_opts=level)


def write_hdf5_tree(
    output_path: Path | str,
    tree: OutputTree,
    *,
    compression_defaults: bool = False,
) -> None:
    """Write a nested output tree to an HDF5 file."""

    with h5py.File(str(output_path), "w") as fh:
        _write_hdf5_mapping(fh, tree, compression_defaults=compression_defaults)


def write_matlab_tree(
    output_path: Path | str,
    tree: OutputTree,
    *,
    root_name: str = "data",
) -> None:
    """Write a nested output tree to a MATLAB .mat file."""

    savemat(
        str(output_path),
        {root_name: _to_matlab_value(tree)},
        do_compression=True,
        long_field_names=True,
        oned_as="column",
    )


def _write_hdf5_mapping(
    parent: h5py.Group,
    mapping: Mapping[str, Any],
    *,
    compression_defaults: bool,
) -> None:
    for raw_key, raw_value in mapping.items():
        node = raw_value if isinstance(raw_value, OutputNode) else OutputNode(raw_value)
        if node.value is None and node.skip_if_none:
            continue
        key = str(raw_key)
        _write_hdf5_value(parent, key, node, compression_defaults=compression_defaults)


def _write_hdf5_value(
    parent: h5py.Group,
    key: str,
    node: OutputNode,
    *,
    compression_defaults: bool,
) -> None:
    value = node.value
    if isinstance(value, Mapping):
        group = parent.create_group(key)
        for attr_key, attr_value in node.attrs.items():
            group.attrs[str(attr_key)] = attr_value
        _write_hdf5_mapping(group, value, compression_defaults=compression_defaults)
        return

    data, dtype = _coerce_hdf5_data(value, dtype=node.dtype)
    kwargs: dict[str, Any] = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    compression = node.compression
    compression_opts = node.compression_opts
    if compression is None and compression_defaults and _should_compress(data):
        compression = "gzip"
        compression_opts = 4
    if compression is not None and _can_compress(data):
        kwargs["compression"] = compression
        if compression_opts is not None:
            kwargs["compression_opts"] = compression_opts
    ds = parent.create_dataset(key, data=data, **kwargs)
    for attr_key, attr_value in node.attrs.items():
        ds.attrs[str(attr_key)] = attr_value


def _coerce_hdf5_data(value: Any, *, dtype: Any | None) -> tuple[Any, Any | None]:
    str_dtype = h5py.string_dtype(encoding="utf-8")
    if value is None:
        return np.array([], dtype=np.float64), dtype
    if isinstance(value, str):
        return value, dtype or str_dtype
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace"), dtype or str_dtype
    if isinstance(value, np.ndarray):
        if dtype is not None:
            return value, dtype
        if value.dtype.kind in {"U", "S", "O"} and _array_is_string_like(value):
            return value.astype(object), str_dtype
        return value, None
    if isinstance(value, (list, tuple)):
        arr = np.asarray(value, dtype=object)
        if _array_is_string_like(arr):
            return arr, dtype or str_dtype
        try:
            numeric = np.asarray(value)
        except ValueError:
            return arr, dtype
        return numeric, dtype
    if isinstance(value, np.generic):
        return value.item(), dtype
    return value, dtype


def _array_is_string_like(values: np.ndarray) -> bool:
    if values.dtype.kind in {"U", "S"}:
        return True
    if values.dtype.kind != "O":
        return False
    for item in values.ravel():
        if item is None:
            continue
        if not isinstance(item, (str, bytes, np.str_, np.bytes_)):
            return False
    return True


def _can_compress(data: Any) -> bool:
    return isinstance(data, np.ndarray) and data.shape != ()


def _should_compress(data: Any) -> bool:
    return isinstance(data, np.ndarray) and data.size > 1024 and data.dtype.kind in "fiu"


def _to_matlab_value(value: Any) -> Any:
    if isinstance(value, OutputNode):
        if value.value is None and value.skip_if_none:
            return None
        return _to_matlab_value(value.value)
    if isinstance(value, Mapping):
        fields: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            node = raw_value if isinstance(raw_value, OutputNode) else OutputNode(raw_value)
            if node.value is None and node.skip_if_none:
                continue
            name = node.matlab_name or matlab_safe_name(str(raw_key))
            mat_value = _to_matlab_value(node.value)
            if mat_value is not None:
                fields[name] = mat_value
        return make_struct(**fields)
    if isinstance(value, tuple):
        value = list(value)
    if isinstance(value, list):
        if _is_ragged_sequence(value):
            out = np.empty(len(value), dtype=object)
            for idx, item in enumerate(value):
                out[idx] = _to_matlab_value(item)
            return out
        return np.asarray(value, dtype=object if _sequence_is_string_like(value) else None)
    if isinstance(value, str):
        return np.str_(value)
    if isinstance(value, bytes):
        return np.str_(value.decode("utf-8", errors="replace"))
    if isinstance(value, bool):
        return bool(value)
    return value


def _sequence_is_string_like(values: list[Any]) -> bool:
    return all(isinstance(item, (str, bytes, np.str_, np.bytes_)) for item in values)


def _is_ragged_sequence(values: list[Any]) -> bool:
    if not values:
        return False
    if any(isinstance(item, Mapping) for item in values):
        return True
    if any(isinstance(item, (list, tuple)) for item in values):
        shapes = [np.asarray(item).shape for item in values]
        return len(set(shapes)) > 1
    if any(isinstance(item, np.ndarray) for item in values):
        shapes = [np.asarray(item).shape for item in values]
        return len(set(shapes)) > 1
    return False

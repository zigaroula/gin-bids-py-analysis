"""Helpers for writing and reading MATLAB .mat files from processing results.

MATLAB struct layout conventions
---------------------------------
``scipy.io.savemat`` converts a 1-element numpy structured array (whose dtype
is ``[(field_name, object)]``) into a MATLAB struct.  ``make_struct`` builds
exactly this representation so that callers simply nest calls to produce a
hierarchy that mirrors the HDF5 group layout of each writer.

Field-name sanitization
-----------------------
MATLAB field names are limited to 63 characters, must start with a letter, and
may contain only letters, digits, and underscores.  Condition labels supplied
by users (e.g. ``"condition_a"``) are used as struct field names in the
``means`` and ``uncertainty`` groups; ``matlab_safe_name`` ensures they are
always valid regardless of what the user provided.

Reading helpers
---------------
When loading .mat files with ``scipy.io.loadmat(..., squeeze_me=True,
struct_as_record=False)`` the returned values do not always have predictable
Python types — a single-element string array becomes a bare ``str``, a
zero-element array stays an ``ndarray``, scalar numerics may be 0-d arrays or
plain Python numbers, etc.  The ``mat_str``, ``mat_float``, ``mat_int``, and
``mat_str_list`` helpers normalise these corner-cases so callers can extract
values defensively without knowing the exact squeeze shape.
"""

from __future__ import annotations

import math
import re
from typing import Any

import numpy as np


def matlab_round(value: float) -> int:
    """Round like MATLAB: halves are rounded away from zero.

    Python's built-in ``round`` uses bankers rounding, so values exactly at
    ``.5`` can differ from MATLAB by one integer.
    """
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def matlab_safe_name(name: str) -> str:
    """Return a MATLAB-safe struct field name derived from *name*.

    Rules applied in order:
    1. Collapse any run of characters that are not ``[a-zA-Z0-9_]`` into ``_``.
    2. Strip leading/trailing underscores (artifacts of step 1).
    3. If the result starts with a digit, prefix with ``f_``.
    4. Truncate to 63 characters (MATLAB's ``namelengthmax``).
    5. If the result is empty after all transforms, return ``field``.
    """
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    safe = safe.strip("_")
    if safe and safe[0].isdigit():
        safe = "f_" + safe
    safe = safe[:63]
    return safe or "field"


def make_struct(**fields: object) -> np.ndarray:
    """Build a 1-element numpy structured array that ``scipy.io.savemat`` turns into a MATLAB struct.

    Each keyword argument becomes one field in the resulting struct.  Values
    may be numpy arrays, plain Python scalars, or nested structured arrays
    returned by another :func:`make_struct` call.

    Example::

        axes = make_struct(
            time_s=np.array([0.0, 0.1, 0.2]),
            channel=np.array(["A1", "A2"], dtype=object),
        )
        top = make_struct(stats=stats_struct, axes=axes)
        scipy.io.savemat("out.mat", {"data": top})
    """
    if not fields:
        # Return an empty struct placeholder.
        dt = np.dtype([("_empty", object)])
        arr = np.zeros(1, dtype=dt)
        arr["_empty"][0] = np.array([], dtype=object)
        return arr

    dt = np.dtype([(k, object) for k in fields])
    arr = np.zeros(1, dtype=dt)
    for key, value in fields.items():
        arr[key][0] = value
    return arr


def mat_str(value: Any, *, default: str = "") -> str:
    """Extract a plain Python string from a value returned by ``loadmat``.

    Handles the squeeze_me=True cases:
    - Plain ``str`` or ``bytes`` → decoded directly.
    - 0-d numpy array → element extracted then decoded.
    - 1-D or N-D numpy array → first element used (scalar case after squeeze).
    - ``None`` or anything else → *default* returned.
    """
    if value is None:
        return default
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        el = value.flat[0]
        if isinstance(el, bytes):
            return el.decode("utf-8", errors="replace")
        return str(el)
    try:
        return str(value)
    except Exception:
        return default


def mat_float(value: Any, *, default: float) -> float:
    """Extract a plain Python float from a value returned by ``loadmat``."""
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return float(value.flat[0])
    try:
        return float(value)
    except Exception:
        return default


def mat_int(value: Any, *, default: int) -> int:
    """Extract a plain Python int from a value returned by ``loadmat``."""
    if value is None:
        return default
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return default
        return int(value.flat[0])
    try:
        return int(value)
    except Exception:
        return default


def mat_str_list(value: Any) -> list[str]:
    """Extract a list of strings from a value returned by ``loadmat``.

    Handles the squeeze_me=True edge cases:
    - ``None`` or missing → empty list.
    - Bare ``str`` or ``bytes`` (1-element array squeezed to scalar) → ``[decoded]``.
    - Empty ndarray → ``[]``.
    - 1-D or N-D ndarray of str/bytes/object → decoded elements in flat order.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, bytes):
        return [value.decode("utf-8", errors="replace")]
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return []
        result: list[str] = []
        for el in value.flat:
            if isinstance(el, bytes):
                result.append(el.decode("utf-8", errors="replace"))
            else:
                result.append(str(el))
        return result
    return []


def matlab_tukeywin(N, r):
    if r <= 0:
        return np.ones(N, dtype=np.float64)
    elif r >= 1:
        n = np.arange(N, dtype=np.float64)
        return 0.5*(1 - np.cos(2*np.pi*n/(N-1)))
    else:
        w = np.ones(N, dtype=np.float64)
        edge = int(np.floor(r*(N-1)/2.0))
        n = np.arange(0, edge+1, dtype=np.float64)
        w[:edge+1] = 0.5*(1 + np.cos(np.pi*(2*n/(r*(N-1)) - 1)))
        n = np.arange(N-edge-1, N, dtype=np.float64)
        w[N-edge-1:] = 0.5*(1 + np.cos(np.pi*(2*(n-(N-1))/(r*(N-1)) + 1)))
        return w

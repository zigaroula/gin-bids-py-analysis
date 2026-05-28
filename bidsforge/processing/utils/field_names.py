"""Helpers for sanitizing and validating strings used as HDF5 group keys and MATLAB struct field names."""

from __future__ import annotations

import re

_VALID_FIELD_CHARS_RE = re.compile(r"[^a-zA-Z0-9_]")


def sanitize_field_name(name: str) -> str:
    """Replace characters forbidden in MATLAB struct field names and HDF5 group keys with underscores."""
    return _VALID_FIELD_CHARS_RE.sub("_", name)


def validate_field_name(original: str, sanitized: str, *, context: str = "Field") -> None:
    """Raise ValueError if *sanitized* is not a valid MATLAB identifier.

    Rules enforced:
    - Must not start with a digit.
    - Must not exceed 63 characters (MATLAB namelengthmax).
    """
    if sanitized[0].isdigit():
        raise ValueError(
            f"{context} {original!r} starts with a digit after sanitization: {sanitized!r}. "
            "Please rename it to start with a letter or underscore."
        )
    if len(sanitized) > 63:
        raise ValueError(
            f"{context} {original!r} has {len(sanitized)} characters after sanitization "
            "(MATLAB namelengthmax is 63). Please use a shorter name."
        )

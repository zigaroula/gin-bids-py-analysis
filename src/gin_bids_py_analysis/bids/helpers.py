from __future__ import annotations

from pathlib import Path

# Canonical BIDS entity ordering (determines the order of components in a filename).
# See: https://bids-specification.readthedocs.io/en/stable/appendices/entity-table.html
_ENTITY_ORDER = [
    "sub", "tpl", "ses", "cohort", "sample", "task", "tracksys", "acq",
    "nuc", "voi", "ce", "trc", "stain", "rec", "dir", "run", "mod", "echo",
    "flip", "inv", "mt", "part", "proc", "hemi", "space", "split", "recording",
    "chunk", "atlas", "seg", "scale", "res", "den", "label", "desc"
]


def build_bids_path(
    entities: dict[str, str],
    root: Path,
    suffix: str,
    extension: str,
    datatype: str | None = None,
) -> Path:
    """
    Construct a BIDS-compliant file path from entity key/value pairs.

    Entities are written in canonical BIDS order; any extra entities not in the
    canonical list are appended alphabetically at the end.

    Args:
        entities:   Mapping of BIDS entity keys to values
                    (e.g. ``{"sub": "01", "run": "1"}``).
                    ``"sub"``/``"subject"`` and ``"ses"``/``"session"`` are
                    both accepted as aliases.
        root:       Dataset root or derivatives pipeline root directory.
        suffix:     BIDS suffix (e.g. ``"ieeg"``, ``"hilbert"``).
        extension:  File extension **including** the leading dot (e.g. ``".npy"``).
        datatype:   BIDS datatype sub-folder (e.g. ``"ieeg"``).
                    Defaults to *suffix* when not provided.

    Returns:
        Absolute :class:`~pathlib.Path` for the output file.
    """
    # Normalise subject/session aliases so both "sub"/"subject" and "ses"/"session" work
    norm: dict[str, str] = {}
    for k, v in entities.items():
        if k == "subject":
            norm["sub"] = v
        elif k == "session":
            norm["ses"] = v
        else:
            norm[k] = v

    # Build filename components in canonical BIDS entity order
    parts: list[str] = []
    for key in _ENTITY_ORDER:
        if key in norm:
            parts.append(f"{key}-{norm[key]}")

    filename = "_".join(parts) + f"_{suffix}{extension}"

    # Folder: root / sub-<id> / [ses-<id> /] <datatype> /
    sub = norm.get("sub", "unknown")
    ses = norm.get("ses")
    resolved_datatype = datatype or suffix

    folder = root / f"sub-{sub}"
    if ses:
        folder = folder / f"ses-{ses}"
    folder = folder / resolved_datatype

    return folder / filename


def parse_entities(path: Path) -> dict[str, str]:
    """
    Parse BIDS entities from a filename stem.

    Returns a dict of entity key/value pairs (e.g. ``{"sub": "01", "task": "rest"}``).
    Components without a ``"-"`` (i.e. the suffix) are ignored.

    Note: ``suffix`` and ``extension`` are **not** included here; use
    :attr:`BIDSFile.suffix` and :attr:`BIDSFile.extension` for those.
    """
    stem = path.stem
    entities: dict[str, str] = {}
    for part in stem.split("_"):
        if "-" in part:
            key, _, value = part.partition("-")
            entities[key] = value
    return entities

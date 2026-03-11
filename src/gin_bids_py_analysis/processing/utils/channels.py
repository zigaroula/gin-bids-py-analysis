"""Channel-montage utilities shared across processing subpackages."""

from __future__ import annotations

from collections import defaultdict
from enum import Enum

import numpy as np


class MontageMode(str, Enum):
    """How channels are referenced before analysis."""

    MONO = "mono"
    """Each channel is used directly, without re-referencing."""

    BIPOLAR = "bipolar"
    """Adjacent pairs are subtracted; pairs are detected automatically from
    channel names (electrode prefix + contact index, e.g. ``A1``, ``A2``)."""


class BipolarDirection(str, Enum):
    """Which contact is subtracted from which in a bipolar pair."""

    NEXT_MINUS_PREVIOUS = "next_minus_previous"
    """channel[i+1] - channel[i]  (e.g. A2 - A1)"""

    PREVIOUS_MINUS_NEXT = "previous_minus_next"
    """channel[i] - channel[i+1]  (e.g. A1 - A2)"""


class BipolarStorage(str, Enum):
    """Which channel name is used as the label for the derived bipolar signal."""

    PREVIOUS = "previous"
    """Label the result with the lower-index contact name (e.g. ``A1``)."""

    NEXT = "next"
    """Label the result with the higher-index contact name (e.g. ``A2``)."""

    PREVIOUS_MINUS_NEXT = "previous_minus_next"
    """Label as ``A1-A2`` (lower - higher convention)."""

    NEXT_MINUS_PREVIOUS = "next_minus_previous"
    """Label as ``A2-A1`` (higher - lower convention.)"""


def _parse_channel_name(name: str) -> tuple[str, int] | None:
    """Split a channel name into (electrode_prefix, contact_index).

    Assumes the convention ``<letters><digits>`` (e.g. ``"A1"`` → ``("A", 1)``).
    Returns ``None`` if the name does not end with digits.

    Args:
        name: Raw channel name string.

    Returns:
        ``(prefix, index)`` tuple or ``None``.
    """
    # Find where the trailing digits begin
    split = len(name)
    while split > 0 and name[split - 1].isdigit():
        split -= 1
    if split == len(name) or split == 0:
        # No trailing digits or all digits → cannot parse
        return None
    prefix = name[:split]
    index = int(name[split:])
    return prefix, index


def build_montage(
    data: np.ndarray,
    channel_names: list[str],
    mode: MontageMode,
    direction: BipolarDirection,
    storage: BipolarStorage,
) -> tuple[np.ndarray, list[str]]:
    """Apply a channel montage to *data*.

    Mono mode
    ~~~~~~~~~
    Returns *data* and *channel_names* unchanged.

    Bipolar mode
    ~~~~~~~~~~~~
    Channels are grouped by electrode prefix (letters before the trailing
    number, e.g. ``"A"`` for ``"A1"``, ``"A2"``).  Within each group,
    adjacent contacts (consecutive integers) form a pair.  The subtraction
    direction and output label are controlled by *direction* and *storage*.

    Channels whose names cannot be parsed or that have no adjacent neighbour
    are silently dropped.

    Args:
        data:          2-D array of shape ``[n_channels, n_samples]``.
        channel_names: List of channel name strings, same order as *data* rows.
        mode:          :class:`MontageMode` — mono or bipolar.
        direction:     :class:`BipolarDirection` — which contact is subtracted.
        storage:       :class:`BipolarStorage` — naming convention for the output.

    Returns:
        ``(montaged_data, montaged_names)`` where *montaged_data* has shape
        ``[n_montaged_channels, n_samples]``.
    """
    if mode == MontageMode.MONO:
        return data, list(channel_names)

    # ------------------------------------------------------------------
    # Bipolar: group channels by electrode prefix, sort by contact index
    # ------------------------------------------------------------------
    # Map from electrode prefix → sorted list of (contact_index, row_index)
    electrode_groups: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row_idx, name in enumerate(channel_names):
        parsed = _parse_channel_name(name)
        if parsed is None:
            continue  # non-parseable channels are skipped
        prefix, contact_idx = parsed
        electrode_groups[prefix].append((contact_idx, row_idx))

    out_rows: list[np.ndarray] = []
    out_names: list[str] = []

    for prefix in sorted(electrode_groups):
        contacts = sorted(electrode_groups[prefix])  # sorted by contact index
        for i in range(len(contacts) - 1):
            prev_idx, prev_row = contacts[i]
            next_idx, next_row = contacts[i + 1]

            # Only pair truly adjacent contacts (no skips)
            if next_idx != prev_idx + 1:
                continue

            prev_sig = data[prev_row]
            next_sig = data[next_row]
            prev_name = channel_names[prev_row]
            next_name = channel_names[next_row]

            if direction == BipolarDirection.NEXT_MINUS_PREVIOUS:
                diff = next_sig - prev_sig
            else:  # PREVIOUS_MINUS_NEXT
                diff = prev_sig - next_sig

            # Choose the output channel label
            if storage == BipolarStorage.PREVIOUS:
                label = prev_name
            elif storage == BipolarStorage.NEXT:
                label = next_name
            elif storage == BipolarStorage.PREVIOUS_MINUS_NEXT:
                label = f"{prev_name}-{next_name}"
            else:  # NEXT_MINUS_PREVIOUS
                label = f"{next_name}-{prev_name}"

            out_rows.append(diff)
            out_names.append(label)

    if not out_rows:
        raise ValueError(
            "Bipolar montage produced no channels — check that channel names "
            "follow the '<letters><digits>' convention (e.g. 'A1', 'A2')."
        )

    montaged_data = np.stack(out_rows, axis=0)
    return montaged_data, out_names

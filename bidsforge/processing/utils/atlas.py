"""Atlas-based channel grouping utilities shared across processing pipelines.

Provides helpers to map iEEG channels to atlas regions from an electrodes TSV/CSV
and to aggregate epoch data by those regions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import mne
import numpy as np

from bidsforge.bids.file import BIDSFile
from bidsforge.bids.matching import find_best_entity_match
from bidsforge.processing.utils.channels import normalize_channel_name
from bidsforge.processing.utils.tables import select_column


@dataclass(frozen=True)
class AtlasGrouping:
    """Intermediate result of mapping iEEG channels to atlas regions for one file."""

    source_file: BIDSFile              # Electrodes TSV/CSV the mapping was built from.
    feature_names: list[str]           # Ordered list of atlas region names.
    feature_channel_indices: list[np.ndarray]  # Per-region channel row indices into the data matrix.
    missing_regions: list[str]         # Requested regions absent from the electrodes table.


def resolve_atlas_grouping(
    *,
    ieeg_file: BIDSFile,
    electrodes_files: Sequence[BIDSFile],
    channel_names: Sequence[str],
    atlas_name: str,
    atlas_regions: Sequence[str],
) -> AtlasGrouping:
    """Build an ``AtlasGrouping`` for *ieeg_file* from the best-matching electrodes table.

    Steps:
    1. Find the electrodes file whose entities best match the iEEG file.
    2. Parse the table to build a channel→region mapping using the *atlas_name* column.
    3. Group iEEG channel indices by region; optionally filter to *atlas_regions*.
    """
    # Step 1: find the best-matching electrodes file.
    electrodes_file = find_best_entity_match(
        ieeg_file,
        electrodes_files,
        ambiguity_label="electrodes table",
        ambiguity_hint=(
            "Please disambiguate entities (e.g. run/task) in *_electrodes.tsv."
        ),
    )
    if electrodes_file is None:
        raise ValueError(
            f"atlas_name={atlas_name!r} requires a matching *_electrodes.tsv/csv file "
            f"for {ieeg_file.path.name}."
        )

    # Step 2: parse the electrodes table and build the channel→region mapping.
    with electrodes_file.ensure_loaded() as rows:
        if not rows:
            raise ValueError(f"Electrodes table {electrodes_file.path.name} is empty.")

        columns = list(rows[0].keys())
        channel_col = select_column(columns, preferred=["name", "channel", "label"])
        atlas_col = select_column(columns, preferred=[atlas_name])
        if channel_col is None:
            raise ValueError(
                f"Electrodes table {electrodes_file.path.name} must contain a channel name "
                "column (e.g. 'name')."
            )
        if atlas_col is None:
            raise ValueError(
                f"Electrodes table {electrodes_file.path.name} has no column matching "
                f"atlas_name={atlas_name!r}."
            )

        channel_to_region: dict[str, str] = {}
        for row in rows:
            raw_channel = (row.get(channel_col) or "").strip()
            raw_region = (row.get(atlas_col) or "").strip()
            if not raw_channel or not raw_region:
                continue
            channel_key = normalize_channel_name(raw_channel)
            previous = channel_to_region.get(channel_key)
            if previous is not None and previous != raw_region:
                raise ValueError(
                    f"Channel {raw_channel!r} has multiple atlas labels in "
                    f"{electrodes_file.path.name}: {previous!r} and {raw_region!r}."
                )
            channel_to_region[channel_key] = raw_region

        if not channel_to_region:
            raise ValueError(
                f"No non-empty atlas labels found in column {atlas_col!r} of "
                f"{electrodes_file.path.name}."
            )

    # Step 3: group iEEG channel row indices by atlas region.
    region_to_indices: dict[str, list[int]] = {}
    for idx, channel_name in enumerate(channel_names):
        channel_key = normalize_channel_name(channel_name)
        region = channel_to_region.get(channel_key)
        if region is None:
            continue
        region_to_indices.setdefault(region, []).append(idx)

    if not region_to_indices:
        raise ValueError(
            f"No ieeg channels from {ieeg_file.path.name} matched atlas labels in "
            f"{electrodes_file.path.name}."
        )

    # Filter to the requested subset of regions (if any), collecting missing ones.
    missing_regions: list[str] = []
    if atlas_regions:
        available_by_norm = {
            region.casefold(): region for region in region_to_indices
        }
        selected_regions: list[str] = []
        for requested in atlas_regions:
            region = available_by_norm.get(requested.casefold())
            if region is None:
                missing_regions.append(requested)
                continue
            if region not in selected_regions:
                selected_regions.append(region)
        if not selected_regions:
            available = ", ".join(sorted(region_to_indices.keys()))
            raise ValueError(
                f"None of atlas_regions={list(atlas_regions)!r} were found in "
                f"{electrodes_file.path.name}. Available regions: {available}"
            )
    else:
        selected_regions = [
            region
            for region in region_to_indices
            if not _is_na_like_region_label(region)
        ]
        if not selected_regions:
            available = ", ".join(sorted(region_to_indices.keys()))
            raise ValueError(
                "No usable atlas regions remained after removing n/a-like labels in "
                f"{electrodes_file.path.name}. Available regions: {available}"
            )

    feature_indices = [
        np.asarray(region_to_indices[region], dtype=np.int64)
        for region in selected_regions
    ]

    return AtlasGrouping(
        source_file=electrodes_file,
        feature_names=selected_regions,
        feature_channel_indices=feature_indices,
        missing_regions=missing_regions,
    )


def aggregate_epochs_with_mne(
    epochs: np.ndarray,
    *,
    channel_names: Sequence[str],
    feature_names: Sequence[str],
    feature_channel_indices: Sequence[np.ndarray],
    sfreq: float,
    tmin_s: float,
) -> np.ndarray:
    """Aggregate channels into ROI means via ``mne.channels.combine_channels``.

    Input shape:  (n_epochs, n_channels, n_times)
    Output shape: (n_epochs, n_features, n_times)
    """
    if epochs.ndim != 3:
        raise ValueError(
            f"epochs must be 3-D (n_epochs, n_channels, n_times), got {epochs.shape!r}."
        )
    if epochs.shape[0] == 0:
        return np.empty((0, len(feature_names), epochs.shape[2]), dtype=np.float64)

    info = mne.create_info(
        ch_names=list(channel_names),
        sfreq=float(sfreq),
        ch_types=["seeg"] * len(channel_names),
    )
    events = np.column_stack(
        [
            np.arange(epochs.shape[0], dtype=np.int64),
            np.zeros(epochs.shape[0], dtype=np.int64),
            np.ones(epochs.shape[0], dtype=np.int64),
        ]
    )
    epochs_obj = mne.EpochsArray(
        np.asarray(epochs, dtype=np.float64),
        info=info,
        events=events,
        event_id={"anchor": 1},
        tmin=float(tmin_s),
        verbose=False,
    )
    groups = {
        str(name): np.asarray(indices, dtype=np.int64).tolist()
        for name, indices in zip(feature_names, feature_channel_indices)
    }
    combined = mne.channels.combine_channels(
        epochs_obj,
        groups=groups,
        method="mean",
        keep_stim=False,
        drop_bad=False,
    )
    return np.asarray(combined.get_data(copy=True), dtype=np.float64)


def _is_na_like_region_label(label: str) -> bool:
    """Return True for common ``n/a`` placeholder variants."""
    normalized = "".join(ch for ch in label.strip().casefold() if ch.isalnum())
    return normalized in {"na", "notavailable", "notapplicable"}

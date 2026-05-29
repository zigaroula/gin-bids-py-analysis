"""
HFO/spike detector analysis — run script.
Edit the parameters below and run: python scripts/debug/run_hfo_spike_detection.py
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

from bidsforge.bids import BIDSDataset
from bidsforge.bids.helpers import normalize_subject_value
from bidsforge.processing.hfo_spike_detection import (
    HfoSpikeDetectorParams,
    HfoSpikeDetectorProcessing,
    HfoSpikeDetectorProcessingWriter,
    HfoSpikeDetectorWriterParams,
)
from bidsforge.processing.utils.channels import (
    BipolarDirection,
    BipolarStorage,
    MontageMode,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

BIDS_ROOT = Path(r"E:\Boulot\clarissa_bids")

# BIDS entity filters: only files matching ALL of these will be processed.
# Remove any key you don't want to filter on.
FILE_FILTERS = {
    "suffix": "ieeg",
    "extension": ".vhdr"
    #"run": "01",
}

# Optional: CSV files with two columns (subject, channel) that define
# per-subject channel selection for the montage.  Channels from all files
# are merged into a single subject → channel-list mapping.
# Set to an empty dict (or remove entries) to disable and use all channels.
CHANNELS_CSV_FILES = {
    "vmPFC": Path(r"E:\Boulot\csv\PFCvm_elecs_tbl.csv"),
    "daINS": Path(r"E:\Boulot\csv\aINS_dors_elecs_tbl.csv"),
    "vaINS": Path(r"E:\Boulot\csv\aINS_vent_elecs_tbl.csv"),
}

# Algorithm parameters for detection
PARAMS = HfoSpikeDetectorParams(
    detection_type=["Osc", "Spk"],
    montage_mode=MontageMode.BIPOLAR,
    bipolar_direction=BipolarDirection.NEXT_MINUS_PREVIOUS,
    bipolar_storage=BipolarStorage.NEXT
)

# Writer configuration for output files
WRITER_PARAMS = HfoSpikeDetectorWriterParams(
    bids_root=BIDS_ROOT,
    output_format="matlab"
)

N_JOBS = 1  # parallelism across files; set to -1 to use all available CPUs

# ---------------------------------------------------------------------------
# CSV helpers for per-subject channel selection
# ---------------------------------------------------------------------------

_NA_LIKE_TOKENS = frozenset({"nan", "na", "n/a", "none", "null"})
_FIRST_CONTACT_PATTERN = re.compile(r"^([A-Za-z]+[0-9]+)")
_CONTACT_TOKEN_PATTERN = re.compile(r"[A-Za-z']+[0-9]+")
_PARTICIPANTS_SOURCE_SUBJECT_COLUMNS = (
    "source_subject_id",
    "original_subject_id",
    "original_id",
    "source_id",
)


def _is_nan_like(value: object) -> bool:
    return str(value).strip().casefold() in _NA_LIKE_TOKENS


def _normalize_external_subject_value(subject: object) -> str:
    label = str(subject).strip().replace("_", "")
    return normalize_subject_value(label)


def _subject_lookup_key(subject: object) -> str:
    return _normalize_external_subject_value(subject).casefold()


def _load_source_to_bids_subject_map(bids_root: Path) -> dict[str, str]:
    participants_path = Path(bids_root) / "participants.tsv"
    if not participants_path.exists():
        return {}

    with participants_path.open("r", encoding="utf-8-sig", newline="") as tsv_file:
        reader = csv.DictReader(tsv_file, delimiter="\t")
        if not reader.fieldnames or "participant_id" not in reader.fieldnames:
            return {}

        source_col = next(
            (
                column
                for column in _PARTICIPANTS_SOURCE_SUBJECT_COLUMNS
                if column in reader.fieldnames
            ),
            None,
        )
        if source_col is None:
            return {}

        source_to_bids: dict[str, str] = {}
        for row in reader:
            bids_subject = normalize_subject_value(row.get("participant_id", ""))
            source_subject = _normalize_external_subject_value(row.get(source_col, ""))
            if not bids_subject or not source_subject:
                continue
            source_to_bids[_subject_lookup_key(source_subject)] = bids_subject

        return source_to_bids


def _normalize_subject_from_csv(
    raw_subject: object,
    *,
    source_to_bids: dict[str, str],
) -> str:
    normalized = _normalize_external_subject_value(raw_subject)
    return source_to_bids.get(_subject_lookup_key(normalized), normalized)


def _extract_first_bipolar_contact(raw_channel: object) -> str:
    """Return the first contact from a channel name or bipolar pair label."""
    channel = str(raw_channel).strip()
    if not channel:
        return ""
    token_matches = _CONTACT_TOKEN_PATTERN.findall(channel)
    if token_matches:
        return token_matches[0]
    match = _FIRST_CONTACT_PATTERN.match(channel)
    if match is not None:
        return match.group(1)
    for separator in ("-", "_", " "):
        if separator in channel:
            return channel.split(separator, 1)[0].strip()
    return channel


def _extract_second_bipolar_contact(raw_channel: object) -> str:
    """Return the second contact from a bipolar pair label (e.g. 'Xp01-Xp02' → 'Xp02').

    Returns an empty string when no separator is found (i.e. the input is
    already a single contact name).
    """
    channel = str(raw_channel).strip()
    if not channel:
        return ""
    token_matches = _CONTACT_TOKEN_PATTERN.findall(channel)
    if len(token_matches) >= 2:
        return token_matches[1]
    for separator in ("-", "_", " "):
        if separator in channel:
            return channel.split(separator, 1)[1].strip()
    return ""


def _load_subject_channels_from_csv(
    csv_path: Path,
    *,
    source_to_bids: dict[str, str],
) -> dict[str, list[str]]:
    """Load a subject → channel-list mapping from a two-column CSV.

    For each row the first column is treated as the subject ID and the second
    as a channel name or bipolar pair label (e.g. ``Xp01-Xp02``).  Both
    contacts of a bipolar pair are added so that the bipolar montage can be
    built correctly.  Rows that contain NaN-like values are skipped.
    Duplicate channels within the same subject are removed while preserving
    insertion order.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Mapping of normalised subject IDs to deduplicated channel lists.

    Raises:
        FileNotFoundError: If *csv_path* does not exist.
        ValueError: If the file has fewer than two columns.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Channel CSV not found: {csv_path}")

    subject_channels: dict[str, list[str]] = {}
    seen_channels: dict[str, set[str]] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames or len(reader.fieldnames) < 2:
            raise ValueError(
                f"{csv_path}: expected at least 2 columns (subject, channel)."
            )
        subject_col = reader.fieldnames[0]
        channel_col = reader.fieldnames[1]

        for row in reader:
            raw_subject = row.get(subject_col, "")
            raw_channel_value = row.get(channel_col, "")
            if _is_nan_like(raw_subject) or _is_nan_like(raw_channel_value):
                continue

            subject = _normalize_subject_from_csv(
                raw_subject,
                source_to_bids=source_to_bids,
            )
            raw_channel = str(raw_channel_value).strip()
            if not subject or not raw_channel:
                continue

            # Collect both contacts of a bipolar pair so the montage can be built.
            contacts_to_add = [
                _extract_first_bipolar_contact(raw_channel),
                _extract_second_bipolar_contact(raw_channel),
            ]

            subject_seen = seen_channels.setdefault(subject, set())
            for contact in contacts_to_add:
                if not contact:
                    continue
                key = contact.casefold()
                if key in subject_seen:
                    continue
                subject_seen.add(key)
                subject_channels.setdefault(subject, []).append(contact)

    return subject_channels


def _merge_subject_channels(
    csv_paths_by_roi: dict[str, Path],
) -> dict[str, list[str]]:
    """Load and merge subject → channel-list mappings from multiple CSV files.

    Channels from all CSVs are concatenated per subject.  Duplicates across
    files are removed while preserving insertion order.

    Args:
        csv_paths_by_roi: Mapping of label → CSV path (label is ignored, only
            used for error reporting).

    Returns:
        Merged mapping of normalised subject IDs to deduplicated channel lists.

    Raises:
        FileNotFoundError: If any CSV path does not exist.
        ValueError: If no channels were loaded from any CSV after filtering.
    """
    merged: dict[str, list[str]] = {}
    seen: dict[str, set[str]] = {}
    source_to_bids = _load_source_to_bids_subject_map(BIDS_ROOT)

    for label, csv_path in csv_paths_by_roi.items():
        per_file = _load_subject_channels_from_csv(
            csv_path,
            source_to_bids=source_to_bids,
        )
        for subject, channels in per_file.items():
            subject_seen = seen.setdefault(subject, set())
            for channel in channels:
                key = channel.casefold()
                if key in subject_seen:
                    continue
                subject_seen.add(key)
                merged.setdefault(subject, []).append(channel)
        print(f"CSV {label!r}: loaded {sum(len(v) for v in per_file.values())} contact(s) across {len(per_file)} subject(s).")

    if not merged:
        raise ValueError(
            "No channels were loaded from any CSV file after filtering invalid subject/channel rows."
        )

    return merged


def _filter_files_with_subject_channels(
    files: list,
    channels_by_subject: dict[str, list[str]],
) -> list:
    """Keep only files whose subject has at least one configured channel."""
    kept = []
    skipped_subjects: set[str] = set()

    for file in files:
        subject = file.get("subject") if hasattr(file, "get") else None
        channels = channels_by_subject.get(subject or "")
        if channels:
            kept.append(file)
            continue
        skipped_subjects.add(str(subject or "<unknown>"))

    if skipped_subjects:
        skipped = ", ".join(sorted(skipped_subjects))
        print(
            "Skipping "
            f"{len(files) - len(kept)} file(s) because no selected channel "
            f"was found for subject(s): {skipped}."
        )

    return kept


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    channels_for_montage = None
    if CHANNELS_CSV_FILES:
        channels_for_montage = _merge_subject_channels(CHANNELS_CSV_FILES)
        print(f"Total: {sum(len(v) for v in channels_for_montage.values())} unique contact(s) across {len(channels_for_montage)} subject(s).")
        PARAMS = PARAMS.model_copy(update={"channels_for_montage": channels_for_montage})

    ds = BIDSDataset(BIDS_ROOT)
    files = ds.get_files(scope="raw", **FILE_FILTERS)
    if channels_for_montage is not None:
        files = _filter_files_with_subject_channels(files, channels_for_montage)
    print(f"Found {len(files)} file(s). Running with n_jobs={N_JOBS}.")

    processor = HfoSpikeDetectorProcessing(PARAMS)
    writer = HfoSpikeDetectorProcessingWriter(WRITER_PARAMS)

    out_paths = processor.run(files, writer, n_jobs=N_JOBS, skip_existing=True)
    for p in out_paths:
        print(f"Wrote {p}")


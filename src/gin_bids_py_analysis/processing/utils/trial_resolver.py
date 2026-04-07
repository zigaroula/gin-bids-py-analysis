"""Trial resolution utilities shared across processing pipelines.

Provides the ``TrialResolver`` protocol and a table-driven implementation
(``TableTrialResolver``) that matches iEEG anchor events to trial labels
read from TSV/CSV secondary files carried in a ``BIDSFileGroup``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.bids.matching import entities_compatible
from gin_bids_py_analysis.processing.utils.events import AnnotationEvent
from gin_bids_py_analysis.processing.utils.tables import (
    LoadedTableRow,
    load_table_rows,
    row_float_value,
    row_int_value,
    row_code_value,
    row_value,
)


# String values that are treated as boolean False when reading keep-columns.
_FALSEY = {"0", "false", "f", "no", "n", "off"}


@dataclass
class ResolvedTrial:
    """One anchor event paired with a task-specific trial label."""

    source_file: BIDSFile
    anchor_event_index: int
    anchor_event_code: str | None
    anchor_onset_s: float
    anchor_duration_s: float
    label: str | None = None
    trial_id: str | None = None
    keep: bool = True
    exclusion_reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TrialResolver(Protocol):
    """Interface for task-specific mapping from anchor events to trial labels."""

    def resolve_trials(
        self,
        group: BIDSFileGroup,
        ieeg_file: BIDSFile,
        anchor_events: Sequence[AnnotationEvent],
    ) -> list[ResolvedTrial]:
        """Return one resolved trial per candidate anchor event."""


@dataclass
class _TrialRow:
    """Normalized, file-independent representation of one row from the input table."""

    label: str | None
    trial_id: str | None
    anchor_onset_s: float | None
    anchor_event_order: int | None
    anchor_event_code: str | None
    keep: bool
    exclusion_reason: str | None
    metadata: dict[str, Any]


class TableTrialResolver(BaseModel):
    """Resolve trials from TSV/CSV tables carried in the BIDSFileGroup."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    label_column: str | None = Field(
        default=None,
        description=(
            "Column that carries trial labels. When unset, the resolver builds trials from "
            "anchor rows only and leaves labels unset."
        ),
    )
    label_map: dict[str, str] = Field(
        default_factory=dict,
        description="Optional mapping used to canonicalize label values (case-insensitive keys).",
    )
    extract_columns: list[str] = Field(
        default_factory=list,
        description=(
            "Optional list of column names copied into resolved trial metadata. Values are read "
            "from label rows with event-row fallback."
        ),
    )
    trial_id_column: str | None = Field(
        default="trial_id",
        description="Optional trial identifier column used to join event and label rows.",
    )
    anchor_onset_column: str | None = Field(
        default="onset",
        description="Optional anchor onset column used for onset-based matching.",
    )
    anchor_event_code_column: str | None = Field(
        default="anchor_event_code",
        description="Optional anchor event code column used to constrain row/event pairing.",
    )
    anchor_event_order_column: str | None = Field(
        default="anchor_event_order",
        description="Optional anchor order column used for deterministic fallback ordering.",
    )
    keep_column: str | None = Field(
        default=None,
        description="Optional keep flag column. Falsey values mark a trial as excluded.",
    )
    exclusion_reason_column: str | None = Field(
        default=None,
        description="Optional column carrying exclusion reasons.",
    )
    onset_tolerance_s: float = Field(
        default=1e-3,
        ge=0.0,
        description="Maximum onset mismatch (seconds) accepted for onset-based row matching.",
    )

    @field_validator(
        "label_column",
        "trial_id_column",
        "anchor_onset_column",
        "anchor_event_code_column",
        "anchor_event_order_column",
        "keep_column",
        "exclusion_reason_column",
        mode="before",
    )
    @classmethod
    def _normalize_optional_columns(cls, value: object) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @field_validator("label_map", mode="before")
    @classmethod
    def _normalize_label_map(cls, value: object) -> dict[str, str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("label_map must be a dict[str, str].")
        normalized: dict[str, str] = {}
        for raw_key, raw_val in value.items():
            key = str(raw_key).strip().casefold()
            mapped = str(raw_val).strip()
            if key and mapped:
                normalized[key] = mapped
        return normalized

    @field_validator("extract_columns", mode="before")
    @classmethod
    def _normalize_extract_columns(cls, value: object) -> list[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise TypeError("extract_columns must be a list[str].")
        normalized: list[str] = []
        seen_columns: set[str] = set()
        for raw_name in value:
            cleaned = str(raw_name).strip()
            if not cleaned or cleaned in seen_columns:
                continue
            seen_columns.add(cleaned)
            normalized.append(cleaned)
        return normalized

    def resolve_trials(
        self,
        group: BIDSFileGroup,
        ieeg_file: BIDSFile,
        anchor_events: Sequence[AnnotationEvent],
    ) -> list[ResolvedTrial]:
        """Load matching table rows from the group, build trial rows, then match to anchor events."""
        table_rows = load_table_rows(group.secondaries)
        matching_rows = [
            row
            for row in table_rows
            if entities_compatible(
                ieeg_file.entities,
                row.file.entities,
                preferred_entities=row.values,
            )
        ]
        trial_rows = self._build_trial_rows(matching_rows)
        return self._match_anchor_events(ieeg_file, anchor_events, trial_rows)

    def _build_trial_rows(self, rows: Sequence[LoadedTableRow]) -> list[_TrialRow]:
        """Convert raw table rows into ``_TrialRow`` objects, choosing the best available strategy.

        Strategy priority:
        0. No ``label_column`` configured: use anchor rows directly and leave labels unset.
        1. Rows that have both a label *and* anchor information (self-contained).
        2. Separate event rows joined to label rows via trial_id or positional order.
        3. Label-only rows (no anchor info available in the table at all).
        """
        if self.label_column is None:
            anchor_rows = [row for row in rows if self._has_anchor_info(row)]
            if not anchor_rows:
                return []
            ordered_anchor_rows = self._sort_rows(anchor_rows)
            return [
                self._make_trial_row(event_row=row, label_row=None)
                for row in ordered_anchor_rows
            ]

        labeled_rows = [row for row in rows if row_value(row, self.label_column)]
        if not labeled_rows:
            return []

        # Strategy 1: some labeled rows already carry anchor info — use them directly.
        direct_rows = [row for row in labeled_rows if self._has_anchor_info(row)]
        if direct_rows:
            return [self._make_trial_row(event_row=row, label_row=row) for row in direct_rows]

        # Strategy 2: anchor info lives in separate event rows — join them to the label rows.
        event_rows = [row for row in rows if self._has_anchor_info(row)]
        if event_rows:
            joined_rows = self._join_event_and_label_rows(event_rows, labeled_rows)
            if joined_rows:
                return joined_rows

        # Strategy 3: no anchor info at all — rely purely on positional matching later.
        return [self._make_trial_row(event_row=None, label_row=row) for row in labeled_rows]

    def _join_event_and_label_rows(
        self,
        event_rows: Sequence[LoadedTableRow],
        labeled_rows: Sequence[LoadedTableRow],
    ) -> list[_TrialRow]:
        """Join separate event and label rows, first by trial_id, then by positional order."""
        if self.trial_id_column:
            # Preferred: join on shared trial_id values.
            events_by_trial_id = {
                trial_id: row
                for row in event_rows
                if (trial_id := row_value(row, self.trial_id_column)) is not None
            }
            if events_by_trial_id:
                joined = [
                    self._make_trial_row(
                        event_row=events_by_trial_id[trial_id],
                        label_row=label_row,
                    )
                    for label_row in labeled_rows
                    if (trial_id := row_value(label_row, self.trial_id_column)) in events_by_trial_id
                ]
                if joined:
                    return joined

        # Fallback: pair by sorted position (event_order → onset → row_index).
        ordered_events = self._sort_rows(event_rows)
        ordered_labels = self._sort_rows(labeled_rows)
        return [
            self._make_trial_row(event_row=event_row, label_row=label_row)
            for event_row, label_row in zip(ordered_events, ordered_labels)
        ]

    def _match_anchor_events(
        self,
        ieeg_file: BIDSFile,
        anchor_events: Sequence[AnnotationEvent],
        trial_rows: Sequence[_TrialRow],
    ) -> list[ResolvedTrial]:
        """Map each anchor event to a trial row and build the final ``ResolvedTrial`` list.

        Two-phase strategy:
        1. Onset-based pre-matching: greedily assign the closest unmatched row whose
           onset falls within ``onset_tolerance_s`` of the anchor event.
        2. Sequential fallback: for events without an onset match, consume rows in
           sorted order (event_order → onset → row_index), filtered by event code.
        """
        resolved: list[ResolvedTrial] = []
        unmatched_rows = set(range(len(trial_rows)))
        matched_rows_by_event: dict[int, int] = {}  # anchor_index → trial_row_index

        # Phase 1: onset-based pre-matching (only when the table carries onset values).
        if any(row.anchor_onset_s is not None for row in trial_rows):
            for anchor_index, anchor_event in enumerate(anchor_events):
                candidates = [
                    row_index
                    for row_index in unmatched_rows
                    if self._row_matches_anchor_onset(trial_rows[row_index], anchor_event)
                ]
                if not candidates:
                    continue
                best_row_index = min(
                    candidates,
                    key=lambda row_index: abs(
                        float(trial_rows[row_index].anchor_onset_s) - anchor_event.onset_s
                    ),
                )
                matched_rows_by_event[anchor_index] = best_row_index
                unmatched_rows.remove(best_row_index)

        # Phase 2: build an ordered iterator over still-unmatched rows for sequential fallback.
        ordered_rows = [
            row_index
            for row_index in self._ordered_trial_row_indices(trial_rows)
            if row_index in unmatched_rows
        ]
        ordered_iter = iter(ordered_rows)

        # Assign one trial row to each anchor event, preferring onset-matched rows.
        for anchor_index, anchor_event in enumerate(anchor_events):
            row_index = matched_rows_by_event.get(anchor_index)
            if row_index is None:
                # No onset match — consume the next compatible row sequentially.
                row_index = self._next_matching_order_row(
                    ordered_iter,
                    trial_rows,
                    anchor_event,
                )

            if row_index is None:
                resolved.append(
                    ResolvedTrial(
                        source_file=ieeg_file,
                        anchor_event_index=anchor_index,
                        anchor_event_code=anchor_event.code,
                        anchor_onset_s=anchor_event.onset_s,
                        anchor_duration_s=anchor_event.duration_s,
                        keep=False,
                        exclusion_reason="no_matching_table_row",
                    )
                )
                continue

            trial_row = trial_rows[row_index]
            keep = trial_row.keep
            exclusion_reason = trial_row.exclusion_reason
            if trial_row.label is None:
                keep = False
                exclusion_reason = exclusion_reason or "missing_label"

            resolved.append(
                ResolvedTrial(
                    source_file=ieeg_file,
                    anchor_event_index=anchor_index,
                    anchor_event_code=anchor_event.code,
                    anchor_onset_s=anchor_event.onset_s,
                    anchor_duration_s=anchor_event.duration_s,
                    label=trial_row.label,
                    trial_id=trial_row.trial_id,
                    keep=keep,
                    exclusion_reason=exclusion_reason,
                    metadata=dict(trial_row.metadata),
                )
            )

        return resolved

    def _ordered_trial_row_indices(self, rows: Sequence[_TrialRow]) -> list[int]:
        """Return row indices sorted by event_order, then onset, then original index."""
        return sorted(
            range(len(rows)),
            key=lambda idx: (
                rows[idx].anchor_event_order
                if rows[idx].anchor_event_order is not None
                else 10**9,
                rows[idx].anchor_onset_s
                if rows[idx].anchor_onset_s is not None
                else float("inf"),
                idx,
            ),
        )

    def _next_matching_order_row(
        self,
        ordered_iter: Iterator[int],
        trial_rows: Sequence[_TrialRow],
        anchor_event: AnnotationEvent,
    ) -> int | None:
        """Consume the iterator until a row whose event_code matches (or is unset) is found."""
        for row_index in ordered_iter:
            row = trial_rows[row_index]
            if row.anchor_event_code is None or row.anchor_event_code == anchor_event.code:
                return row_index
        return None

    def _row_matches_anchor_onset(
        self,
        row: _TrialRow,
        anchor_event: AnnotationEvent,
    ) -> bool:
        """Return True if the row's onset and optional event code are compatible with the anchor."""
        if row.anchor_onset_s is None:
            return False
        if row.anchor_event_code is not None and row.anchor_event_code != anchor_event.code:
            return False
        return abs(row.anchor_onset_s - anchor_event.onset_s) <= self.onset_tolerance_s

    def _make_trial_row(
        self,
        *,
        event_row: LoadedTableRow | None,
        label_row: LoadedTableRow | None,
    ) -> _TrialRow:
        """Build a ``_TrialRow`` by merging anchor info from *event_row* and label from *label_row*.

        For each field the event_row value takes precedence; the label_row is used as fallback.
        """
        trial_id = row_value(event_row, self.trial_id_column) or row_value(label_row, self.trial_id_column)
        anchor_onset_s = row_float_value(event_row, self.anchor_onset_column)
        if anchor_onset_s is None:
            anchor_onset_s = row_float_value(label_row, self.anchor_onset_column)

        anchor_event_order = row_int_value(event_row, self.anchor_event_order_column)
        if anchor_event_order is None:
            anchor_event_order = row_int_value(label_row, self.anchor_event_order_column)

        anchor_event_code = row_code_value(event_row, self.anchor_event_code_column)
        if anchor_event_code is None:
            anchor_event_code = row_code_value(label_row, self.anchor_event_code_column)

        keep = self._keep_value(label_row)
        if event_row is not None:
            keep = keep and self._keep_value(event_row)

        metadata: dict[str, Any] = {}
        if label_row is not None:
            metadata["label_source_path"] = str(label_row.file.path)
            metadata["label_row_index"] = label_row.row_index
            metadata["label_raw"] = row_value(label_row, self.label_column) or ""
        else:
            metadata["label_raw"] = ""
        for column_name in self.extract_columns:
            value = row_value(label_row, column_name)
            if value is None or value == "":
                value = row_value(event_row, column_name)
            if value is not None:
                metadata[column_name] = value
        if event_row is not None:
            metadata["event_source_path"] = str(event_row.file.path)
            metadata["event_row_index"] = event_row.row_index

        return _TrialRow(
            label=self._canonicalize_label(row_value(label_row, self.label_column)),
            trial_id=trial_id,
            anchor_onset_s=anchor_onset_s,
            anchor_event_order=anchor_event_order,
            anchor_event_code=anchor_event_code,
            keep=keep,
            exclusion_reason=row_value(label_row, self.exclusion_reason_column)
            or row_value(event_row, self.exclusion_reason_column),
            metadata=metadata,
        )

    def _has_anchor_info(self, row: LoadedTableRow) -> bool:
        """Return True if the row carries at least one piece of anchor information."""
        return (
            row_float_value(row, self.anchor_onset_column) is not None
            or row_int_value(row, self.anchor_event_order_column) is not None
            or row_value(row, self.anchor_event_code_column) is not None
        )

    def _keep_value(self, row: LoadedTableRow | None) -> bool:
        """Return True when the keep-column is absent, blank, or not a falsey string."""
        raw_value = row_value(row, self.keep_column)
        if not raw_value:
            return True
        return raw_value.strip().casefold() not in _FALSEY

    def _canonicalize_label(self, raw_label: str | None) -> str | None:
        """Strip whitespace and apply the optional label_map; return None for blank labels."""
        if raw_label is None or raw_label == "":
            return None
        normalized = raw_label.strip()
        mapped = self.label_map.get(normalized.casefold())
        return mapped if mapped is not None else normalized

    def _sort_rows(self, rows: Sequence[LoadedTableRow]) -> list[LoadedTableRow]:
        """Sort table rows by event_order, then onset, then original row_index."""
        return sorted(
            rows,
            key=lambda row: (
                row_int_value(row, self.anchor_event_order_column)
                if self.anchor_event_order_column is not None
                and row_int_value(row, self.anchor_event_order_column) is not None
                else 10**9,
                row_float_value(row, self.anchor_onset_column)
                if self.anchor_onset_column is not None
                and row_float_value(row, self.anchor_onset_column) is not None
                else float("inf"),
                row.row_index,
            ),
        )



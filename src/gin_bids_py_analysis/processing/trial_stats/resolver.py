from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Any, Iterator, Protocol, Sequence

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.processing.utils.events import AnnotationEvent


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


class TrialLabelResolver(Protocol):
    """Interface for task-specific mapping from anchor events to trial labels."""

    def resolve_trials(
        self,
        group: BIDSFileGroup,
        ieeg_file: BIDSFile,
        anchor_events: Sequence[AnnotationEvent],
    ) -> list[ResolvedTrial]:
        """Return one resolved trial per candidate anchor event."""


@dataclass
class _LoadedRow:
    values: dict[str, str]
    file: BIDSFile
    row_index: int


@dataclass
class _TrialRow:
    label: str | None
    trial_id: str | None
    anchor_onset_s: float | None
    anchor_event_order: int | None
    anchor_event_code: str | None
    keep: bool
    exclusion_reason: str | None
    metadata: dict[str, Any]


class TableTrialLabelResolver:
    """Resolve trials from TSV/CSV tables carried in the BIDSFileGroup."""

    def __init__(
        self,
        *,
        label_column: str,
        label_map: dict[str, str] | None = None,
        trial_id_column: str = "trial_id",
        anchor_onset_column: str | None = "onset",
        anchor_event_code_column: str | None = "anchor_event_code",
        anchor_event_order_column: str | None = "anchor_event_order",
        keep_column: str | None = None,
        exclusion_reason_column: str | None = None,
        onset_tolerance_s: float = 1e-3,
    ) -> None:
        self.label_column = label_column
        self.label_map = {
            str(k).strip().casefold(): str(v).strip()
            for k, v in (label_map or {}).items()
        }
        self.trial_id_column = trial_id_column
        self.anchor_onset_column = anchor_onset_column
        self.anchor_event_code_column = anchor_event_code_column
        self.anchor_event_order_column = anchor_event_order_column
        self.keep_column = keep_column
        self.exclusion_reason_column = exclusion_reason_column
        self.onset_tolerance_s = onset_tolerance_s

    def resolve_trials(
        self,
        group: BIDSFileGroup,
        ieeg_file: BIDSFile,
        anchor_events: Sequence[AnnotationEvent],
    ) -> list[ResolvedTrial]:
        table_rows = self._load_table_rows(group.secondaries)
        matching_rows = [
            row
            for row in table_rows
            if _matches_entities(row.file.entities, ieeg_file.entities, row.values)
        ]
        trial_rows = self._build_trial_rows(matching_rows)
        return self._match_anchor_events(ieeg_file, anchor_events, trial_rows)

    def _load_table_rows(self, files: Sequence[BIDSFile]) -> list[_LoadedRow]:
        rows: list[_LoadedRow] = []
        for file in files:
            if file.extension not in {".tsv", ".csv"}:
                continue
            delimiter = "\t" if file.extension == ".tsv" else ","
            with open(file.path, "r", encoding="utf-8", newline="") as fh:
                reader = csv.DictReader(fh, delimiter=delimiter)
                for idx, raw_row in enumerate(reader):
                    values = {
                        str(key).strip(): "" if value is None else str(value).strip()
                        for key, value in raw_row.items()
                    }
                    rows.append(_LoadedRow(values=values, file=file, row_index=idx))
        return rows

    def _build_trial_rows(self, rows: Sequence[_LoadedRow]) -> list[_TrialRow]:
        labeled_rows = [row for row in rows if self._row_value(row, self.label_column)]
        if not labeled_rows:
            return []

        direct_rows = [row for row in labeled_rows if self._has_anchor_info(row)]
        if direct_rows:
            return [self._make_trial_row(event_row=row, label_row=row) for row in direct_rows]

        event_rows = [row for row in rows if self._has_anchor_info(row)]
        if event_rows:
            joined_rows = self._join_event_and_label_rows(event_rows, labeled_rows)
            if joined_rows:
                return joined_rows

        return [self._make_trial_row(event_row=None, label_row=row) for row in labeled_rows]

    def _join_event_and_label_rows(
        self,
        event_rows: Sequence[_LoadedRow],
        labeled_rows: Sequence[_LoadedRow],
    ) -> list[_TrialRow]:
        if self.trial_id_column:
            events_by_trial_id = {
                trial_id: row
                for row in event_rows
                if (trial_id := self._trial_id(row)) is not None
            }
            if events_by_trial_id:
                joined = [
                    self._make_trial_row(
                        event_row=events_by_trial_id[trial_id],
                        label_row=label_row,
                    )
                    for label_row in labeled_rows
                    if (trial_id := self._trial_id(label_row)) in events_by_trial_id
                ]
                if joined:
                    return joined

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
        resolved: list[ResolvedTrial] = []
        unmatched_rows = set(range(len(trial_rows)))
        matched_rows_by_event: dict[int, int] = {}

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

        ordered_rows = [
            row_index
            for row_index in self._ordered_trial_row_indices(trial_rows)
            if row_index in unmatched_rows
        ]
        ordered_iter = iter(ordered_rows)

        for anchor_index, anchor_event in enumerate(anchor_events):
            row_index = matched_rows_by_event.get(anchor_index)
            if row_index is None:
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
        if row.anchor_onset_s is None:
            return False
        if row.anchor_event_code is not None and row.anchor_event_code != anchor_event.code:
            return False
        return abs(row.anchor_onset_s - anchor_event.onset_s) <= self.onset_tolerance_s

    def _make_trial_row(
        self,
        *,
        event_row: _LoadedRow | None,
        label_row: _LoadedRow,
    ) -> _TrialRow:
        trial_id = self._trial_id(event_row) or self._trial_id(label_row)
        anchor_onset_s = self._float_value(event_row, self.anchor_onset_column)
        if anchor_onset_s is None:
            anchor_onset_s = self._float_value(label_row, self.anchor_onset_column)

        anchor_event_order = self._int_value(event_row, self.anchor_event_order_column)
        if anchor_event_order is None:
            anchor_event_order = self._int_value(label_row, self.anchor_event_order_column)

        anchor_event_code = self._code_value(event_row)
        if anchor_event_code is None:
            anchor_event_code = self._code_value(label_row)

        keep = self._keep_value(label_row)
        if event_row is not None:
            keep = keep and self._keep_value(event_row)

        metadata: dict[str, Any] = {
            "label_source_path": str(label_row.file.path),
            "label_row_index": label_row.row_index,
            "label_raw": self._row_value(label_row, self.label_column) or "",
        }
        if event_row is not None:
            metadata["event_source_path"] = str(event_row.file.path)
            metadata["event_row_index"] = event_row.row_index

        return _TrialRow(
            label=self._canonicalize_label(self._row_value(label_row, self.label_column)),
            trial_id=trial_id,
            anchor_onset_s=anchor_onset_s,
            anchor_event_order=anchor_event_order,
            anchor_event_code=anchor_event_code,
            keep=keep,
            exclusion_reason=self._row_value(label_row, self.exclusion_reason_column)
            or self._row_value(event_row, self.exclusion_reason_column),
            metadata=metadata,
        )

    def _has_anchor_info(self, row: _LoadedRow) -> bool:
        return (
            self._float_value(row, self.anchor_onset_column) is not None
            or self._int_value(row, self.anchor_event_order_column) is not None
            or self._code_value(row) is not None
        )

    def _keep_value(self, row: _LoadedRow | None) -> bool:
        raw_value = self._row_value(row, self.keep_column)
        if not raw_value:
            return True
        return raw_value.strip().casefold() not in _FALSEY

    def _canonicalize_label(self, raw_label: str | None) -> str | None:
        if raw_label is None or raw_label == "":
            return None
        normalized = raw_label.strip()
        mapped = self.label_map.get(normalized.casefold())
        return mapped if mapped is not None else normalized

    def _trial_id(self, row: _LoadedRow | None) -> str | None:
        return self._row_value(row, self.trial_id_column)

    def _code_value(self, row: _LoadedRow | None) -> str | None:
        raw_value = self._row_value(row, self.anchor_event_code_column)
        if raw_value is None or raw_value == "":
            return None
        return str(int(raw_value)) if raw_value.isdigit() else raw_value

    def _float_value(self, row: _LoadedRow | None, column: str | None) -> float | None:
        raw_value = self._row_value(row, column)
        if raw_value in (None, ""):
            return None
        return float(raw_value)

    def _int_value(self, row: _LoadedRow | None, column: str | None) -> int | None:
        raw_value = self._row_value(row, column)
        if raw_value in (None, ""):
            return None
        return int(raw_value)

    def _row_value(self, row: _LoadedRow | None, column: str | None) -> str | None:
        if row is None or column is None:
            return None
        return row.values.get(column)

    def _sort_rows(self, rows: Sequence[_LoadedRow]) -> list[_LoadedRow]:
        return sorted(
            rows,
            key=lambda row: (
                self._int_value(row, self.anchor_event_order_column)
                if self.anchor_event_order_column is not None
                and self._int_value(row, self.anchor_event_order_column) is not None
                else 10**9,
                self._float_value(row, self.anchor_onset_column)
                if self.anchor_onset_column is not None
                and self._float_value(row, self.anchor_onset_column) is not None
                else float("inf"),
                row.row_index,
            ),
        )


def _matches_entities(
    row_file_entities: dict[str, Any],
    target_entities: dict[str, Any],
    row_values: dict[str, str],
) -> bool:
    for canonical_name, aliases in {
        "subject": ("subject", "sub"),
        "session": ("session", "ses"),
        "run": ("run",),
        "task": ("task",),
    }.items():
        target_value = _normalize_entity_value(target_entities.get(canonical_name))
        if target_value is None:
            continue

        row_value = None
        for alias in aliases:
            row_value = _normalize_entity_value(row_values.get(alias))
            if row_value is not None:
                break
            row_value = _normalize_entity_value(row_file_entities.get(alias))
            if row_value is not None:
                break

        if row_value is not None and row_value != target_value:
            return False

    return True


def _normalize_entity_value(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.startswith("sub-"):
        return text[4:]
    if text.startswith("ses-"):
        return text[4:]
    return text

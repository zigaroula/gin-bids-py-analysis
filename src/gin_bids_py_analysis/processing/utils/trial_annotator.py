"""Trial window annotators for enriching and invalidating resolved trials.

Two classes implement the ``TrialWindowAnnotator`` protocol and run in a
shared list between trial resolution and normalization:

* ``EventFileWindowAnnotator`` — pure annotation.  Loads TSV/CSV event files
  from ``group.secondaries`` and stores, for each resolved trial, the list of
  all event row dicts whose onset falls inside the trial's epoch window.  It
  never modifies ``keep`` or ``exclusion_reason``.

* ``EventAnnotationInvalidationRule`` — pure invalidation.  Reads the list
  stored by an annotator (or any other source) from ``trial.metadata``,
  optionally filters it with a :class:`ConditionExpr`, and sets
  ``trial.keep = False`` when the count of matching events reaches the
  configured threshold.

Both classes are valid ``TrialWindowAnnotator`` implementations and can be
mixed freely in the same annotators list.
"""

from __future__ import annotations

from typing import Any, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gin_bids_py_analysis.bids.file import BIDSFile
from gin_bids_py_analysis.bids.file_group import BIDSFileGroup
from gin_bids_py_analysis.bids.matching import files_matching_entities
from gin_bids_py_analysis.processing.utils.condition_rules import (
    ConditionExpr,
    matches_condition_expr,
)
from gin_bids_py_analysis.processing.utils.tables import load_table_rows
from gin_bids_py_analysis.processing.utils.trial_resolver import ResolvedTrial


class TrialWindowAnnotator:
    """Protocol for objects that annotate or invalidate resolved trials.

    Implementations must provide ``annotate_trials()``.  The method mutates
    *trials* in-place; it must not return a value.

    The method receives the full epoch window defaults (``tmin_s``,
    ``tmax_s``) that the processor will use, together with the iEEG channel
    names available at annotation time.  Implementations may ignore any
    argument they do not need.
    """

    def annotate_trials(
        self,
        group: BIDSFileGroup,
        ieeg_file: BIDSFile,
        trials: list[ResolvedTrial],
        tmin_s: float,
        tmax_s: float,
        *,
        ieeg_channel_names: list[str],
    ) -> None:
        raise NotImplementedError


class EventFileWindowAnnotator(BaseModel):
    """Load external event files and annotate each trial with in-window events.

    For every resolved trial the annotator scans the rows of every matched
    event file and stores, under ``trial.metadata[metadata_events_key]``, the
    list of row value dicts whose ``onset_column`` falls inside the trial's
    epoch window ``[anchor_onset_s + tmin, anchor_onset_s + tmax]``.

    The list is *always* written (as ``[]`` when no events match) so that
    downstream :class:`EventAnnotationInvalidationRule` instances can rely on
    the key being present.

    No filtering is applied here.  The annotator collects all events in the
    window regardless of their content.  Filtering and invalidation is the
    responsibility of :class:`EventAnnotationInvalidationRule`.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    filter: dict[str, Any] = Field(
        description=(
            "BIDS entity filter used to select files from ``group.secondaries``. "
            "Passed directly to ``files_matching_entities()`` as keyword arguments. "
            "For example ``{'suffix': 'events', 'desc': 'delphos'}`` selects only "
            "Delphos detection output files."
        ),
    )
    metadata_events_key: str = Field(
        description=(
            "Key written into ``trial.metadata`` that holds the list of matching "
            "event row dicts.  Choose a unique name per annotator so that multiple "
            "annotators can coexist without collision."
        ),
    )
    onset_column: str = Field(
        default="onset",
        description="Column name in the event TSV/CSV that carries onset times in seconds.",
    )
    window_tmin_s: float | None = Field(
        default=None,
        description=(
            "Start of the event search window relative to anchor onset (seconds). "
            "When ``None`` the processor's ``tmin_s`` is used."
        ),
    )
    window_tmax_s: float | None = Field(
        default=None,
        description=(
            "End of the event search window relative to anchor onset (seconds). "
            "When ``None`` the processor's ``tmax_s`` is used."
        ),
    )

    @field_validator("filter", mode="before")
    @classmethod
    def _normalize_filter(cls, value: object) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError("filter must be a dict of BIDS entity key-value pairs.")
        return dict(value)

    @field_validator("metadata_events_key", mode="before")
    @classmethod
    def _normalize_metadata_events_key(cls, value: object) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("metadata_events_key must be a non-empty string.")
        return cleaned

    @field_validator("onset_column", mode="before")
    @classmethod
    def _normalize_onset_column(cls, value: object) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("onset_column must be a non-empty string.")
        return cleaned

    def annotate_trials(
        self,
        group: BIDSFileGroup,
        ieeg_file: BIDSFile,
        trials: list[ResolvedTrial],
        tmin_s: float,
        tmax_s: float,
        *,
        ieeg_channel_names: list[str],
    ) -> None:
        """Write in-window event lists into each trial's metadata dict."""
        del ieeg_file, ieeg_channel_names  # not used by this annotator

        effective_tmin = self.window_tmin_s if self.window_tmin_s is not None else tmin_s
        effective_tmax = self.window_tmax_s if self.window_tmax_s is not None else tmax_s

        matched_files = files_matching_entities(group.secondaries, **self.filter)
        rows = load_table_rows(matched_files)

        # Pre-parse onsets once for all rows; skip rows with missing / invalid onset.
        parsed: list[tuple[float, dict[str, str]]] = []
        for row in rows:
            raw_onset = row.values.get(self.onset_column)
            if raw_onset is None or raw_onset == "":
                continue
            try:
                onset = float(raw_onset)
            except (ValueError, TypeError):
                continue
            parsed.append((onset, dict(row.values)))

        for trial in trials:
            win_start = trial.anchor_onset_s + effective_tmin
            win_end = trial.anchor_onset_s + effective_tmax
            in_window = [
                values
                for onset, values in parsed
                if win_start <= onset <= win_end
            ]
            trial.metadata[self.metadata_events_key] = in_window


class EventAnnotationInvalidationRule(BaseModel):
    """Invalidate trials based on annotated event lists.

    Reads ``trial.metadata[metadata_events_key]`` (populated by an
    :class:`EventFileWindowAnnotator`), applies an optional
    :class:`ConditionExpr` filter to each event dict, and sets
    ``trial.keep = False`` when the number of matching events is at or above
    ``invalidate_if_count_gte``.

    The ``exclusion_reason`` is only set when the trial is being invalidated
    *for the first time*; an already-excluded trial's reason is preserved.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    metadata_events_key: str = Field(
        description=(
            "Metadata key to read from each trial.  Must match the "
            "``metadata_events_key`` of a preceding ``EventFileWindowAnnotator``."
        ),
    )
    event_filter: ConditionExpr | None = Field(
        default=None,
        description=(
            "Optional boolean expression evaluated on each event dict before counting. "
            "When ``None`` every event in the list counts. "
            "Example: ``{'all': [{'column': 'event_type', 'op': '==', 'value': 'Spk'}, "
            "{'column': 'channel', 'op': 'in', 'values': ['vmPFC-1', 'vmPFC-2']}]}``"
        ),
    )
    invalidate_if_count_gte: int = Field(
        default=1,
        ge=1,
        description="Invalidate the trial when the count of matching events reaches this threshold.",
    )
    exclusion_reason: str = Field(
        default="event_in_window",
        description="Exclusion reason written to ``trial.exclusion_reason`` upon invalidation.",
    )

    @field_validator("metadata_events_key", mode="before")
    @classmethod
    def _normalize_metadata_events_key(cls, value: object) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("metadata_events_key must be a non-empty string.")
        return cleaned

    @field_validator("exclusion_reason", mode="before")
    @classmethod
    def _normalize_exclusion_reason(cls, value: object) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("exclusion_reason must be a non-empty string.")
        return cleaned

    def annotate_trials(
        self,
        group: BIDSFileGroup,
        ieeg_file: BIDSFile,
        trials: list[ResolvedTrial],
        tmin_s: float,
        tmax_s: float,
        *,
        ieeg_channel_names: list[str],
    ) -> None:
        """Apply the invalidation rule to each trial in-place."""
        del group, ieeg_file, tmin_s, tmax_s, ieeg_channel_names  # not used

        for trial in trials:
            events: list[dict[str, Any]] = trial.metadata.get(self.metadata_events_key, [])

            if self.event_filter is not None:
                matching = [
                    event
                    for event in events
                    if matches_condition_expr(self.event_filter, event)
                ]
            else:
                matching = list(events)

            if len(matching) >= self.invalidate_if_count_gte:
                trial.keep = False
                if trial.exclusion_reason is None:
                    trial.exclusion_reason = self.exclusion_reason

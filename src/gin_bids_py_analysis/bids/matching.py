from __future__ import annotations

from typing import Any, Mapping, Sequence

from .file import BIDSFile

EntityAliasGroup = tuple[str, ...]

DEFAULT_ENTITY_ALIAS_GROUPS: tuple[EntityAliasGroup, ...] = (
    ("subject", "sub"),
    ("session", "ses"),
    ("task",),
    ("run",),
)

_DEFAULT_IGNORED_SPECIFICITY_ENTITIES = frozenset(
    {"suffix", "extension", "datatype", "scope"}
)
_DEFAULT_EXCLUDED_SHARED_ENTITIES = frozenset(
    {"suffix", "extension", "datatype", "desc", "description"}
)


def normalize_entity_value(value: Any) -> str | None:
    """Normalize BIDS-like entity values for robust comparisons."""
    if value in (None, ""):
        return None
    text = str(value).strip()
    if text.startswith("sub-"):
        return text[4:]
    if text.startswith("ses-"):
        return text[4:]
    return text


def entity_value(entities: Mapping[str, Any], aliases: Sequence[str]) -> str | None:
    """Return the first non-empty normalized entity value among *aliases*."""
    for alias in aliases:
        value = normalize_entity_value(entities.get(alias))
        if value is not None:
            return value
    return None


def entity_match_score(
    target_entities: Mapping[str, Any],
    candidate_entities: Mapping[str, Any],
    *,
    alias_groups: Sequence[Sequence[str]] = DEFAULT_ENTITY_ALIAS_GROUPS,
) -> int | None:
    """
    Return how many explicit entities match between *target* and *candidate*.

    If a candidate explicitly defines an entity that conflicts with the target,
    return ``None``.
    """
    score = 0
    for aliases in alias_groups:
        target = entity_value(target_entities, aliases)
        if target is None:
            continue

        candidate = entity_value(candidate_entities, aliases)
        if candidate is None:
            continue
        if candidate != target:
            return None
        score += 1

    return score


def entities_compatible(
    target_entities: Mapping[str, Any],
    candidate_entities: Mapping[str, Any],
    *,
    preferred_entities: Mapping[str, Any] | None = None,
    alias_groups: Sequence[Sequence[str]] = DEFAULT_ENTITY_ALIAS_GROUPS,
) -> bool:
    """
    Check whether candidate entities are compatible with target entities.

    ``preferred_entities`` can override candidate entities for matching
    (e.g. row-level values should win over file-level entities).
    """
    for aliases in alias_groups:
        target = entity_value(target_entities, aliases)
        if target is None:
            continue

        candidate = (
            entity_value(preferred_entities, aliases)
            if preferred_entities is not None
            else None
        )
        if candidate is None:
            candidate = entity_value(candidate_entities, aliases)

        if candidate is not None and candidate != target:
            return False

    return True


def entity_specificity(
    entities: Mapping[str, Any],
    *,
    ignored_entities: set[str] | frozenset[str] = _DEFAULT_IGNORED_SPECIFICITY_ENTITIES,
) -> int:
    """
    Return an entity specificity score for tie-breaking file matches.

    Lower values mean a file is more generic (fewer explicit entities).
    """
    return sum(
        1
        for key, value in entities.items()
        if key not in ignored_entities and normalize_entity_value(value) is not None
    )


def files_matching_entities(
    files: Sequence[BIDSFile],
    **entity_criteria: Any,
) -> list[BIDSFile]:
    """
    Return files matching all provided entity criteria.

    Criteria values can be scalars (exact match) or a list/tuple/set of
    acceptable values. Values are compared as strings, except ``None`` which
    remains ``None``.
    """
    matched: list[BIDSFile] = []
    for file in files:
        is_match = True
        for key, criterion in entity_criteria.items():
            actual_value = file.get(key)
            normalized_actual = None if actual_value is None else str(actual_value)
            allowed = _criterion_values(criterion)
            if normalized_actual not in allowed:
                is_match = False
                break
        if is_match:
            matched.append(file)
    return sorted(matched, key=lambda file: str(file.path))


def shared_entities(
    files: Sequence[BIDSFile],
    *,
    excluded_entities: set[str] | frozenset[str] = _DEFAULT_EXCLUDED_SHARED_ENTITIES,
) -> dict[str, str]:
    """Return entities shared by all files, excluding provenance-only fields."""
    if not files:
        return {}

    shared = dict(files[0].entities)
    for file in files[1:]:
        shared = {
            key: value
            for key, value in shared.items()
            if file.get(key) == value
        }

    for removable in excluded_entities:
        shared.pop(removable, None)

    return {str(key): str(value) for key, value in shared.items()}


def find_best_entity_match(
    target_file: BIDSFile,
    candidate_files: Sequence[BIDSFile],
    *,
    alias_groups: Sequence[Sequence[str]] = DEFAULT_ENTITY_ALIAS_GROUPS,
    ambiguity_label: str = "file",
    ambiguity_hint: str | None = None,
) -> BIDSFile | None:
    """
    Find the best matching candidate file for *target_file* by BIDS entities.

    Matching favors:
    1. highest entity-match score
    2. lowest entity specificity (more generic file)
    3. lexicographic path tie-break
    """
    scored: list[tuple[int, int, str, BIDSFile]] = []
    for candidate in candidate_files:
        score = entity_match_score(
            target_file.entities,
            candidate.entities,
            alias_groups=alias_groups,
        )
        if score is None:
            continue
        specificity = entity_specificity(candidate.entities)
        scored.append((score, specificity, str(candidate.path), candidate))

    if not scored:
        return None

    best_score = max(score for score, _, _, _ in scored)
    best = [
        (specificity, path, file)
        for score, specificity, path, file in scored
        if score == best_score
    ]
    if len(best) > 1:
        min_specificity = min(specificity for specificity, _, _ in best)
        best = [
            (specificity, path, file)
            for specificity, path, file in best
            if specificity == min_specificity
        ]

    if len(best) > 1:
        names = ", ".join(sorted(file.path.name for _, _, file in best))
        message = (
            f"Ambiguous {ambiguity_label} match for {target_file.path.name}: {names}."
        )
        if ambiguity_hint:
            message = f"{message} {ambiguity_hint}"
        raise ValueError(message)

    return best[0][2]


def _criterion_values(criterion: Any) -> set[str | None]:
    if isinstance(criterion, (list, tuple, set, frozenset)):
        return {None if value is None else str(value) for value in criterion}
    return {None if criterion is None else str(criterion)}

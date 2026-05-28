"""Typed condition rules used to resolve binary trial labels from table rows."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_ORDERED_OPS = frozenset({"<", "<=", ">", ">=", "between"})
_LEAF_OPS = frozenset({"==", "!=", "<", "<=", ">", ">=", "in", "not_in", "between"})
_TRUTHY = frozenset({"1", "true", "t", "yes", "y", "on"})
_FALSEY = frozenset({"0", "false", "f", "no", "n", "off"})
_MATCH_FAILURE_PRIORITY = (
    "condition_value_cast_error",
    "missing_condition_value",
    "missing_condition_column",
    "no_matching_condition",
)


@dataclass(frozen=True)
class ConditionEvaluation:
    """Outcome of evaluating one condition expression against one row."""

    matched: bool
    issues: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ConditionResolution:
    """Resolved label for one row plus audit information."""

    label: str | None
    matched_labels: tuple[str, ...]
    reason: str
    issues: tuple[str, ...]


def _is_number_like(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _coerce_to_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if _is_number_like(value):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("non-finite boolean value")
        if numeric == 0.0:
            return False
        if numeric == 1.0:
            return True
        raise ValueError("numeric boolean values must be 0 or 1")

    text = str(value).strip().casefold()
    if text in _TRUTHY:
        return True
    if text in _FALSEY:
        return False
    raise ValueError(f"Cannot interpret {value!r} as bool.")


def _coerce_to_float(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("bool is not a valid float condition value")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("non-finite float value")
    return numeric


def _coerce_to_int(value: object) -> int:
    if isinstance(value, bool):
        raise ValueError("bool is not a valid int condition value")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value) or not value.is_integer():
            raise ValueError("float must be finite and integer-valued")
        return int(value)

    text = str(value).strip()
    if not text:
        raise ValueError("blank int value")
    try:
        return int(text, 10)
    except ValueError:
        numeric = float(text)
        if not math.isfinite(numeric) or not numeric.is_integer():
            raise ValueError("string must encode an integer-valued number")
        return int(numeric)


def _coerce_to_str(value: object) -> str:
    return str(value).strip()


def _normalize_string(value: str, *, case_sensitive: bool) -> str:
    return value if case_sensitive else value.casefold()


def _normalize_values_list(value: object) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    raise TypeError("values must be a list or tuple.")


def _normalize_bool_field(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return _coerce_to_bool(value)


class ConditionExpr(BaseModel):
    """Recursive boolean expression used to assign a trial to a condition."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
    )

    column: str | None = Field(
        default=None,
        description="Column read from the merged label/event row view.",
    )
    op: Literal["==", "!=", "<", "<=", ">", ">=", "in", "not_in", "between"] | None = Field(
        default=None,
        description="Comparison operator for leaf expressions.",
    )
    value: Any | None = Field(
        default=None,
        description="Single comparison value for scalar operators.",
    )
    values: list[Any] = Field(
        default_factory=list,
        description="Comparison values for set/range operators.",
    )
    cast: Literal["auto", "str", "int", "float", "bool"] = Field(
        default="auto",
        description="Value casting strategy applied before evaluating the operator.",
    )
    case_sensitive: bool = Field(
        default=False,
        description="When False, string comparisons are case-insensitive.",
    )
    all: list[ConditionExpr] = Field(
        default_factory=list,
        description="Logical AND over child expressions.",
    )
    any: list[ConditionExpr] = Field(
        default_factory=list,
        description="Logical OR over child expressions.",
    )
    not_: ConditionExpr | None = Field(
        default=None,
        alias="not",
        description="Logical NOT over one child expression.",
    )

    @field_validator("column", mode="before")
    @classmethod
    def _normalize_column(cls, value: object) -> str | None:
        if value is None:
            return None
        cleaned = str(value).strip()
        return cleaned or None

    @field_validator("values", mode="before")
    @classmethod
    def _coerce_values(cls, value: object) -> list[Any]:
        return _normalize_values_list(value)

    @field_validator("case_sensitive", mode="before")
    @classmethod
    def _coerce_case_sensitive(cls, value: object) -> bool:
        return _normalize_bool_field(value)

    @model_validator(mode="after")
    def _validate_shape(self) -> ConditionExpr:
        has_group = bool(self.all or self.any or self.not_ is not None)
        has_leaf = (
            self.column is not None
            or self.op is not None
            or self.value is not None
            or bool(self.values)
        )

        if has_group and has_leaf:
            raise ValueError(
                "ConditionExpr must define either a leaf comparison or a logical group, not both."
            )
        if not has_group and not has_leaf:
            raise ValueError("ConditionExpr cannot be empty.")

        if has_group:
            active_groups = sum(
                int(flag)
                for flag in (bool(self.all), bool(self.any), self.not_ is not None)
            )
            if active_groups != 1:
                raise ValueError(
                    "ConditionExpr logical groups must define exactly one of 'all', 'any', or 'not'."
                )
            return self

        if self.column is None:
            raise ValueError("Leaf condition expressions require a non-empty column.")
        if self.op is None:
            raise ValueError("Leaf condition expressions require an operator.")
        if self.op not in _LEAF_OPS:
            raise ValueError(f"Unsupported condition operator: {self.op!r}.")

        if self.op in {"in", "not_in", "between"}:
            if self.value is not None:
                raise ValueError(f"Operator {self.op!r} does not use 'value'; use 'values'.")
            if not self.values:
                raise ValueError(f"Operator {self.op!r} requires a non-empty 'values' list.")
            if self.op == "between" and len(self.values) != 2:
                raise ValueError("Operator 'between' requires exactly 2 values.")
        else:
            if self.value is None:
                raise ValueError(f"Operator {self.op!r} requires 'value'.")
            if self.values:
                raise ValueError(f"Operator {self.op!r} does not use 'values'.")

        resolved_cast = self._resolved_cast()
        if self.op in _ORDERED_OPS and resolved_cast not in {"int", "float"}:
            raise ValueError(
                f"Operator {self.op!r} requires numeric values; use numbers or cast='float'/'int'."
            )
        self._coerce_expected_values()
        return self

    def referenced_columns(self) -> set[str]:
        """Return the set of table columns referenced by this expression tree."""
        if self.all:
            return set().union(*(expr.referenced_columns() for expr in self.all))
        if self.any:
            return set().union(*(expr.referenced_columns() for expr in self.any))
        if self.not_ is not None:
            return self.not_.referenced_columns()
        return {self.column} if self.column is not None else set()

    def evaluate(self, row_values: Mapping[str, Any]) -> ConditionEvaluation:
        """Evaluate this expression against one merged table row."""
        if self.all:
            outcomes = [expr.evaluate(row_values) for expr in self.all]
            return ConditionEvaluation(
                matched=all(outcome.matched for outcome in outcomes),
                issues=frozenset().union(*(outcome.issues for outcome in outcomes)),
            )
        if self.any:
            outcomes = [expr.evaluate(row_values) for expr in self.any]
            return ConditionEvaluation(
                matched=any(outcome.matched for outcome in outcomes),
                issues=frozenset().union(*(outcome.issues for outcome in outcomes)),
            )
        if self.not_ is not None:
            child = self.not_.evaluate(row_values)
            if child.issues:
                return ConditionEvaluation(matched=False, issues=child.issues)
            return ConditionEvaluation(matched=not child.matched)

        assert self.column is not None
        if self.column not in row_values:
            return ConditionEvaluation(False, frozenset({"missing_condition_column"}))

        raw_value = row_values[self.column]
        if raw_value is None or (isinstance(raw_value, str) and raw_value.strip() == ""):
            return ConditionEvaluation(False, frozenset({"missing_condition_value"}))

        try:
            actual = self._coerce_runtime_value(raw_value)
        except ValueError:
            return ConditionEvaluation(False, frozenset({"condition_value_cast_error"}))

        expected_values = self._coerce_expected_values()
        matched = self._compare(actual, expected_values)
        return ConditionEvaluation(matched=matched)

    def _compare(self, actual: Any, expected_values: list[Any]) -> bool:
        assert self.op is not None
        if self.op == "==":
            return actual == expected_values[0]
        if self.op == "!=":
            return actual != expected_values[0]
        if self.op == "<":
            return actual < expected_values[0]
        if self.op == "<=":
            return actual <= expected_values[0]
        if self.op == ">":
            return actual > expected_values[0]
        if self.op == ">=":
            return actual >= expected_values[0]
        if self.op == "in":
            return actual in expected_values
        if self.op == "not_in":
            return actual not in expected_values
        if self.op == "between":
            return expected_values[0] <= actual <= expected_values[1]
        raise ValueError(f"Unsupported condition operator: {self.op!r}.")

    def _resolved_cast(self) -> Literal["str", "int", "float", "bool"]:
        if self.cast != "auto":
            return self.cast

        expected_values = self._raw_expected_values()
        if expected_values and all(isinstance(value, bool) for value in expected_values):
            return "bool"
        if expected_values and all(_is_number_like(value) for value in expected_values):
            return "float"
        return "str"

    def _raw_expected_values(self) -> list[Any]:
        if self.op in {"in", "not_in", "between"}:
            return list(self.values)
        if self.value is None:
            return []
        return [self.value]

    def _coerce_expected_values(self) -> list[Any]:
        resolved_cast = self._resolved_cast()
        return [self._coerce_value(value, resolved_cast) for value in self._raw_expected_values()]

    def _coerce_runtime_value(self, value: Any) -> Any:
        return self._coerce_value(value, self._resolved_cast())

    def _coerce_value(
        self,
        value: Any,
        cast: Literal["str", "int", "float", "bool"],
    ) -> Any:
        if cast == "str":
            return _normalize_string(_coerce_to_str(value), case_sensitive=self.case_sensitive)
        if cast == "int":
            return _coerce_to_int(value)
        if cast == "float":
            return _coerce_to_float(value)
        if cast == "bool":
            return _coerce_to_bool(value)
        raise ValueError(f"Unsupported cast mode: {cast!r}.")


class ConditionDefinition(BaseModel):
    """One canonical label plus the expression that assigns trials to it."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    label: str = Field(description="Canonical condition label assigned when the rule matches.")
    when: ConditionExpr = Field(description="Condition expression evaluated on the row.")

    @field_validator("label", mode="before")
    @classmethod
    def _normalize_label(cls, value: object) -> str:
        cleaned = str(value).strip()
        if not cleaned:
            raise ValueError("ConditionDefinition.label must be a non-empty string.")
        return cleaned

    def referenced_columns(self) -> set[str]:
        return self.when.referenced_columns()

    def evaluate(self, row_values: Mapping[str, Any]) -> ConditionEvaluation:
        return self.when.evaluate(row_values)


def resolve_conditions(
    conditions: list[ConditionDefinition],
    row_values: Mapping[str, Any],
) -> ConditionResolution:
    """Resolve one canonical label from many condition definitions."""
    matched_labels: list[str] = []
    issues: set[str] = set()
    for condition in conditions:
        outcome = condition.evaluate(row_values)
        issues.update(outcome.issues)
        if outcome.matched:
            matched_labels.append(condition.label)

    if len(matched_labels) == 1:
        return ConditionResolution(
            label=matched_labels[0],
            matched_labels=tuple(matched_labels),
            reason="matched_condition",
            issues=tuple(sorted(issues)),
        )
    if len(matched_labels) > 1:
        return ConditionResolution(
            label=None,
            matched_labels=tuple(matched_labels),
            reason="ambiguous_condition_match",
            issues=tuple(sorted(issues)),
        )

    return ConditionResolution(
        label=None,
        matched_labels=tuple(),
        reason=_dominant_no_match_reason(issues),
        issues=tuple(sorted(issues)),
    )


def _dominant_no_match_reason(issues: set[str]) -> str:
    for reason in _MATCH_FAILURE_PRIORITY:
        if reason in issues:
            return reason
    return "no_matching_condition"


def matches_condition_expr(expr: ConditionExpr, values: Mapping[str, Any]) -> bool:
    """Return True if *values* satisfies *expr*.

    Evaluates the expression directly without label resolution overhead.
    Issues (missing columns, cast errors) are treated as non-matches.
    """
    outcome = expr.evaluate(values)
    return outcome.matched and not outcome.issues


ConditionExpr.model_rebuild()

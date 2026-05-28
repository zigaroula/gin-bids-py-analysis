from __future__ import annotations

from bidsforge.processing.utils.condition_rules import (
    ConditionDefinition,
    ConditionExpr,
    matches_condition_expr,
    resolve_conditions,
)


def test_condition_expr_supports_case_insensitive_string_equality() -> None:
    expr = ConditionExpr(column="decision", op="==", value="Accept")
    outcome = expr.evaluate({"decision": "accept"})
    assert outcome.matched is True
    assert outcome.issues == frozenset()


def test_condition_expr_supports_membership_operators() -> None:
    expr = ConditionExpr(column="decision", op="in", values=["accept", "maybe"])
    assert expr.evaluate({"decision": "maybe"}).matched is True
    assert expr.evaluate({"decision": "reject"}).matched is False


def test_condition_expr_supports_numeric_thresholds_from_string_inputs() -> None:
    expr = ConditionExpr(column="rating", op=">", value=0)
    assert expr.evaluate({"rating": "3.5"}).matched is True
    assert expr.evaluate({"rating": "-2.0"}).matched is False


def test_condition_expr_supports_inclusive_between() -> None:
    expr = ConditionExpr(column="rating", op="between", values=[-1.0, 1.0])
    assert expr.evaluate({"rating": "-1.0"}).matched is True
    assert expr.evaluate({"rating": "0.25"}).matched is True
    assert expr.evaluate({"rating": "1.5"}).matched is False


def test_condition_expr_supports_bool_cast() -> None:
    expr = ConditionExpr(column="keep", op="==", value=True, cast="bool")
    assert expr.evaluate({"keep": "yes"}).matched is True
    assert expr.evaluate({"keep": "0"}).matched is False


def test_condition_expr_supports_all_any_and_not() -> None:
    expr = ConditionExpr(
        all=[
            {"column": "choice", "op": "==", "value": "1"},
            {
                "any": [
                    {"column": "confidence", "op": ">=", "value": 4},
                    {"not": {"column": "flag", "op": "==", "value": "bad"}},
                ]
            },
        ]
    )

    assert expr.evaluate({"choice": "1", "confidence": "5", "flag": "ok"}).matched is True
    assert expr.evaluate({"choice": "1", "confidence": "2", "flag": "ok"}).matched is True
    assert expr.evaluate({"choice": "1", "confidence": "2", "flag": "bad"}).matched is False


def test_condition_expr_honors_case_sensitive_flag() -> None:
    expr = ConditionExpr(column="decision", op="==", value="Accept", case_sensitive=True)
    assert expr.evaluate({"decision": "Accept"}).matched is True
    assert expr.evaluate({"decision": "accept"}).matched is False


def test_resolve_conditions_reports_missing_value_priority() -> None:
    conditions = [
        ConditionDefinition(label="negative", when={"column": "rating", "op": "<", "value": 0}),
        ConditionDefinition(label="positive", when={"column": "rating", "op": ">", "value": 0}),
    ]

    resolution = resolve_conditions(conditions, {"rating": ""})
    assert resolution.label is None
    assert resolution.reason == "missing_condition_value"


def test_resolve_conditions_reports_ambiguous_matches() -> None:
    conditions = [
        ConditionDefinition(label="negative", when={"column": "rating", "op": "<=", "value": 0}),
        ConditionDefinition(label="positive", when={"column": "rating", "op": ">=", "value": 0}),
    ]

    resolution = resolve_conditions(conditions, {"rating": "0"})
    assert resolution.label is None
    assert resolution.reason == "ambiguous_condition_match"


def test_matches_condition_expr_returns_true_on_match() -> None:
    expr = ConditionExpr(column="event_type", op="==", value="Spk")
    assert matches_condition_expr(expr, {"event_type": "Spk"}) is True


def test_matches_condition_expr_returns_false_on_no_match() -> None:
    expr = ConditionExpr(column="event_type", op="==", value="Spk")
    assert matches_condition_expr(expr, {"event_type": "Osc"}) is False


def test_matches_condition_expr_returns_false_on_missing_column() -> None:
    expr = ConditionExpr(column="event_type", op="==", value="Spk")
    assert matches_condition_expr(expr, {}) is False




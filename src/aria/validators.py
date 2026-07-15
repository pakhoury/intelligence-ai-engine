"""
Deterministic validators that run alongside the LLM reviewer.

These catch issues that don't require LLM judgment:
- SQL safety (forbidden operations, syntax anomalies)
- Result sanity (empty results, suspicious values)
- Answer grounding (claims data but no data was retrieved)
"""
import re


class ValidationResult:
    __slots__ = ("passed", "failures", "score_cap")

    def __init__(self):
        self.passed = True
        self.failures: list[str] = []
        self.score_cap: float = 10.0

    def fail(self, reason: str, cap: float = 5.0):
        self.passed = False
        self.failures.append(reason)
        self.score_cap = min(self.score_cap, cap)


FORBIDDEN_SQL_OPS = re.compile(
    r"\b(DROP|DELETE|TRUNCATE|ALTER|INSERT|UPDATE|CREATE|GRANT|REVOKE|EXEC|EXECUTE)\b",
    re.IGNORECASE,
)

UNBALANCED_PARENS_RE = re.compile(r"[()]")


def validate_sql_safety(sql: str) -> ValidationResult:
    """Check SQL for forbidden operations and basic syntax issues."""
    result = ValidationResult()

    if not sql or sql.strip().upper() == "NO_SQL_POSSIBLE":
        return result

    match = FORBIDDEN_SQL_OPS.search(sql)
    if match:
        result.fail(f"Forbidden SQL operation: {match.group(0).upper()}", cap=0.0)

    opens = sql.count("(")
    closes = sql.count(")")
    if opens != closes:
        result.fail(f"Unbalanced parentheses: {opens} open vs {closes} close", cap=4.0)

    if not re.search(r"\bSELECT\b", sql, re.IGNORECASE):
        result.fail("SQL does not contain SELECT statement", cap=2.0)

    if re.search(r";\s*\w", sql):
        result.fail("Multiple SQL statements detected (possible injection)", cap=0.0)

    return result


def validate_result_sanity(sql_result: str) -> ValidationResult:
    """Sanity-check query results for anomalies."""
    result = ValidationResult()

    if not sql_result or sql_result == "N/A - no SQL data retrieved":
        return result

    if sql_result.startswith(("Error:", "Execution Error:")):
        result.fail(f"SQL execution failed: {sql_result[:100]}", cap=3.0)
        return result

    result_section = sql_result.split("Result:\n", 1)[1] if "Result:\n" in sql_result else sql_result
    if result_section.strip() == "[]":
        result.fail("Query returned empty result set", cap=6.0)

    # Only flag truly absurd magnitudes (>= 10^16, e.g. cartesian-join blowups).
    # Legitimate financial aggregates can reach billions/trillions.
    numbers = re.findall(r"(?<!\w)(\d{16,})(?!\w)", sql_result)
    if numbers:
        result.fail(
            f"Suspiciously large number in result: {numbers[0]} (possible aggregation error)",
            cap=5.0,
        )

    return result


def validate_answer_grounding(
    answer: str,
    sql_result: str,
    retrieved_docs: list[str],
    route: str,
) -> ValidationResult:
    """Check that the answer is grounded in available evidence."""
    result = ValidationResult()

    if not answer:
        result.fail("Empty answer", cap=0.0)
        return result

    has_sql_data = bool(sql_result and not sql_result.startswith(("Error:", "N/A")))
    has_docs = bool(retrieved_docs and any(d.strip() for d in retrieved_docs))

    if (route in ("sql_only", "sql_then_docs", "docs_then_sql", "parallel")
            and not has_sql_data and not has_docs):
        result.fail("Answer generated without any supporting data", cap=3.0)

    number_claims = re.findall(r"\b\d+\.?\d*%?\b", answer)
    if number_claims and not has_sql_data and route != "docs_only":
        result.fail(
            "Answer contains numeric claims but no SQL data was retrieved",
            cap=5.0,
        )

    return result


TRIVIAL_NUMBERS = frozenset({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0, 1000.0})

_NUMBER_TOKEN_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _extract_numeric_values(text: str) -> set:
    """Extract numbers as canonical floats, tolerating thousands separators."""
    values = set()
    for token in _NUMBER_TOKEN_RE.findall(text):
        try:
            values.add(float(token.replace(",", "")))
        except ValueError:
            continue
    return values


def _is_grounded(value: float, sql_numbers: set) -> bool:
    """A claim is grounded if it matches a SQL value exactly or as a rounding of it."""
    for s in sql_numbers:
        if value == s:
            return True
        for decimals in (0, 1, 2):
            if round(s, decimals) == value:
                return True
    return False


def validate_number_grounding(
    answer: str,
    sql_result: str,
) -> ValidationResult:
    """Check that numeric claims in the answer appear in the SQL result.

    Values are compared as canonical floats so that formatting differences
    ("1,234" vs 1234) and reasonable rounding ("55%" from 55.02) don't
    produce false failures.
    """
    result = ValidationResult()

    if not answer or not sql_result or sql_result.startswith(("Error:", "N/A")):
        return result

    answer_numbers = _extract_numeric_values(answer)
    sql_numbers = _extract_numeric_values(sql_result)

    meaningful = answer_numbers - TRIVIAL_NUMBERS
    if not meaningful:
        return result

    ungrounded = [a for a in meaningful if not _is_grounded(a, sql_numbers)]
    if len(ungrounded) > 3:
        examples = ", ".join(f"{u:g}" for u in ungrounded[:3])
        result.fail(
            f"{len(ungrounded)} numeric claims in answer not found in SQL results "
            f"(e.g. {examples})",
            cap=5.0,
        )

    return result


def validate_metric_citation(
    answer: str,
    resolved_metric: str,
    compiled_metric: bool,
) -> ValidationResult:
    """When a metric was compiled deterministically, the answer should reference it."""
    result = ValidationResult()

    if not compiled_metric or not resolved_metric:
        return result

    metric_words = set(resolved_metric.replace("_", " ").lower().split())
    answer_lower = answer.lower()

    matched = sum(1 for w in metric_words if w in answer_lower)
    if matched < len(metric_words) * 0.5:
        result.fail(
            f"Answer does not reference the resolved metric '{resolved_metric}'",
            cap=6.0,
        )

    return result


def run_all_validators(
    sql: str,
    sql_result: str,
    answer: str,
    retrieved_docs: list[str],
    route: str,
    resolved_metric: str = "",
    compiled_metric: bool = False,
) -> tuple[bool, list[str], float]:
    """
    Run all deterministic validators and return aggregate result.

    Returns:
        (all_passed, failure_reasons, score_cap)
    """
    results = [
        validate_sql_safety(sql),
        validate_result_sanity(sql_result),
        validate_answer_grounding(answer, sql_result, retrieved_docs, route),
        validate_number_grounding(answer, sql_result),
        validate_metric_citation(answer, resolved_metric, compiled_metric),
    ]

    all_passed = all(r.passed for r in results)
    failures = []
    score_cap = 10.0

    for r in results:
        failures.extend(r.failures)
        score_cap = min(score_cap, r.score_cap)

    return all_passed, failures, score_cap

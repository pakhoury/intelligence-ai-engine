"""
Deterministic validators that run alongside the LLM reviewer.

These catch issues that don't require LLM judgment:
- SQL safety (forbidden operations, syntax anomalies)
- Result sanity (empty results, suspicious values)
- Answer grounding (claims data but no data was retrieved)
"""
import re

from observability import VALIDATOR_FAILURES


class ValidationResult:
    __slots__ = ("passed", "failures", "score_cap", "unfixable_cap")

    def __init__(self):
        self.passed = True
        self.failures: list[str] = []
        self.score_cap: float = 10.0
        self.unfixable_cap: float = 10.0

    def fail(self, reason: str, cap: float = 5.0, fixable: bool = True):
        """Record a failure. fixable=False marks failures rooted in the
        retrieved data (SQL errors, empty results) rather than the answer
        text — regenerating the answer cannot clear them, so their cap
        re-applies on every reflection cycle.
        """
        self.passed = False
        self.failures.append(reason)
        self.score_cap = min(self.score_cap, cap)
        if not fixable:
            self.unfixable_cap = min(self.unfixable_cap, cap)


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

    # The reflection loop regenerates only the answer, never the SQL, so
    # every SQL-level failure is unfixable by reflection.
    match = FORBIDDEN_SQL_OPS.search(sql)
    if match:
        result.fail(f"Forbidden SQL operation: {match.group(0).upper()}", cap=0.0, fixable=False)

    opens = sql.count("(")
    closes = sql.count(")")
    if opens != closes:
        result.fail(f"Unbalanced parentheses: {opens} open vs {closes} close", cap=4.0, fixable=False)

    if not re.search(r"\bSELECT\b", sql, re.IGNORECASE):
        result.fail("SQL does not contain SELECT statement", cap=2.0, fixable=False)

    if re.search(r";\s*\w", sql):
        result.fail("Multiple SQL statements detected (possible injection)", cap=0.0, fixable=False)

    return result


# Clause keywords that terminate a FROM table list during extraction.
_TABLE_LIST_STOP_WORDS = frozenset({
    "WHERE", "GROUP", "ORDER", "HAVING", "JOIN", "ON", "UNION", "SELECT",
    "FETCH", "LEFT", "RIGHT", "INNER", "OUTER", "FULL", "CROSS", "WITH",
    "LIMIT", "MINUS", "INTERSECT", "CONNECT", "START", "USING",
})

_STRING_LITERAL_RE = re.compile(r"'[^']*'")
_SQL_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_$#]*|[(),]")


def extract_sql_tables(sql: str) -> set[str]:
    """Extract table names referenced after FROM/JOIN, uppercased.

    Handles aliases ("FROM T1 a"), comma-separated FROM lists, and derived
    tables ("FROM (SELECT ...)") — the inner query's own FROM/JOIN keywords
    are scanned on their own, so only real table identifiers are returned.
    """
    tokens = _SQL_TOKEN_RE.findall(_STRING_LITERAL_RE.sub("''", sql))
    tables: set[str] = set()
    i = 0
    while i < len(tokens):
        if tokens[i].upper() not in ("FROM", "JOIN"):
            i += 1
            continue
        j = i + 1
        expecting_table = True
        while j < len(tokens):
            tok = tokens[j]
            upper = tok.upper()
            if tok in ("(", ")") or upper in _TABLE_LIST_STOP_WORDS:
                break
            if tok == ",":
                expecting_table = True
            elif expecting_table:
                tables.add(upper)
                expecting_table = False
            # any other token is an alias — skip it
            j += 1
        i = j if j > i else i + 1
    return tables


def validate_sql_scope(sql: str, allowed_tables: set[str] | None) -> ValidationResult:
    """Check that the SQL references only tables in the governed catalog.

    LLM-generated SQL could otherwise touch any table the database user can
    see. Skipped when no allowlist is available — an empty catalog must not
    block every query.
    """
    result = ValidationResult()

    if not sql or sql.strip().upper() == "NO_SQL_POSSIBLE" or not allowed_tables:
        return result

    allowed = {t.upper() for t in allowed_tables}
    out_of_scope = extract_sql_tables(sql) - allowed
    if out_of_scope:
        result.fail(
            f"SQL references out-of-scope table(s): {', '.join(sorted(out_of_scope))}",
            cap=0.0,
            fixable=False,
        )

    return result


def validate_result_sanity(sql_result: str) -> ValidationResult:
    """Sanity-check query results for anomalies."""
    result = ValidationResult()

    if not sql_result or sql_result == "N/A - no SQL data retrieved":
        return result

    # Result-level failures describe the data, not the answer text —
    # regenerating the answer cannot clear them.
    if sql_result.startswith(("Error:", "Execution Error:")):
        result.fail(f"SQL execution failed: {sql_result[:100]}", cap=3.0, fixable=False)
        return result

    result_section = sql_result.split("Result:\n", 1)[1] if "Result:\n" in sql_result else sql_result
    if result_section.strip() == "[]":
        result.fail("Query returned empty result set", cap=6.0, fixable=False)

    # Only flag truly absurd magnitudes (>= 10^16, e.g. cartesian-join blowups).
    # Legitimate financial aggregates can reach billions/trillions.
    numbers = re.findall(r"(?<!\w)(\d{16,})(?!\w)", sql_result)
    if numbers:
        result.fail(
            f"Suspiciously large number in result: {numbers[0]} (possible aggregation error)",
            cap=5.0,
            fixable=False,
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
        # Data absence, not answer quality — reflection cannot fix it.
        result.fail("Answer generated without any supporting data", cap=3.0, fixable=False)

    # Numeric claims are checked precisely (against SQL results AND retrieved
    # docs, on every route) by validate_number_grounding.
    return result


TRIVIAL_NUMBERS = frozenset({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 100.0, 1000.0})

# Whole numbers in this range are treated as calendar years ("Q1 2024"),
# which legitimately come from the question rather than the data.
YEAR_MIN, YEAR_MAX = 1900, 2100

# Unit abbreviations in answers ("$500K", "$2M") shrink a grounded value by
# one of these factors relative to the raw figure in the data.
_SCALE_FACTORS = (1_000, 1_000_000, 1_000_000_000)

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


def _ratio_values(sql_numbers: set) -> set:
    """Percentages legitimately derivable from the result set: 100*a/b for
    every pair, at the roundings an answer would plausibly quote ("76%" from
    (38, 50)). Only consulted for percent-like claims (0 < value <= 100) so
    coincidental ratios can't accidentally ground large counts.
    """
    ratios = set()
    for a in sql_numbers:
        for b in sql_numbers:
            if not b:
                continue
            ratio = 100.0 * a / b
            for decimals in (0, 1, 2):
                ratios.add(round(ratio, decimals))
    return ratios


def _is_grounded(value: float, source_numbers: set, ratios: set) -> bool:
    """A claim is grounded if it matches a source value exactly, as a
    rounding of it, as a unit-scaled form of it ("$500K" for 500000), or —
    for percent-like values — as a ratio of two source values.
    """
    for s in source_numbers:
        if value == s:
            return True
        for decimals in (0, 1, 2):
            if round(s, decimals) == value:
                return True
    for factor in _SCALE_FACTORS:
        if value * factor in source_numbers:
            return True
    return 0 < value <= 100 and value in ratios


def validate_number_grounding(
    answer: str,
    sql_result: str,
    retrieved_docs: list[str] | None = None,
) -> ValidationResult:
    """Check that every numeric claim in the answer is grounded in the SQL
    result or the retrieved documents. A single unsupported figure fails —
    in a compliance context one hallucinated number is the worst failure
    mode, so there is no tolerance threshold.

    Values are compared as canonical floats so that formatting differences
    ("1,234" vs 1234) and reasonable rounding ("55%" from 55.02) don't
    produce false failures. Also accepted: the total of the result set
    ("totaling 48" from 5+12+23+8), pairwise percentages ("76%" from
    (38, 50)), unit-scaled figures ("$500K" for 500000), and calendar years.
    """
    result = ValidationResult()

    if not answer:
        return result

    answer_numbers = _extract_numeric_values(answer)
    meaningful = {
        v for v in answer_numbers
        if v not in TRIVIAL_NUMBERS
        and not (v.is_integer() and YEAR_MIN <= v <= YEAR_MAX)
    }
    if not meaningful:
        return result

    sql_usable = bool(sql_result) and not sql_result.startswith(
        ("Error:", "Execution Error:", "N/A")
    )
    # Direct matches may come from anywhere in the SQL block (a WHERE-clause
    # threshold is quotable), but derived values (totals, ratios) must be
    # computed from the returned rows only — otherwise date literals in the
    # query text pollute the sum.
    sql_numbers = _extract_numeric_values(sql_result) if sql_usable else set()
    if sql_usable and "Result:\n" in sql_result:
        row_numbers = _extract_numeric_values(sql_result.split("Result:\n", 1)[1])
    else:
        row_numbers = sql_numbers
    doc_numbers = (
        _extract_numeric_values("\n".join(retrieved_docs)) if retrieved_docs else set()
    )

    source_numbers = sql_numbers | doc_numbers
    if row_numbers:
        # "totaling N" over the returned rows is a legitimate derivation.
        source_numbers.add(round(sum(row_numbers), 2))
    ratios = _ratio_values(row_numbers)

    ungrounded = [a for a in meaningful if not _is_grounded(a, source_numbers, ratios)]
    if ungrounded:
        examples = ", ".join(f"{u:g}" for u in sorted(ungrounded)[:3])
        result.fail(
            f"{len(ungrounded)} numeric claim(s) in answer not grounded in SQL "
            f"results or retrieved documents (e.g. {examples})",
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
    allowed_tables: set[str] | None = None,
) -> tuple[bool, list[str], float, float]:
    """
    Run all deterministic validators and return aggregate result.

    Returns:
        (all_passed, failure_reasons, score_cap, unfixable_cap)

    unfixable_cap is the score ceiling imposed by failures rooted in the
    retrieved data rather than the answer text. It re-applies on every
    reflection cycle, so if it is below the review pass threshold the
    reflection loop can never succeed and should be skipped.

    Each validator that fails increments rag_validator_failures_total,
    labeled by validator name, so failure rates are visible on dashboards
    rather than only in text logs.
    """
    named_results = [
        ("sql_safety", validate_sql_safety(sql)),
        ("sql_scope", validate_sql_scope(sql, allowed_tables)),
        ("result_sanity", validate_result_sanity(sql_result)),
        ("answer_grounding", validate_answer_grounding(answer, sql_result, retrieved_docs, route)),
        ("number_grounding", validate_number_grounding(answer, sql_result, retrieved_docs)),
        ("metric_citation", validate_metric_citation(answer, resolved_metric, compiled_metric)),
    ]

    all_passed = True
    failures = []
    score_cap = 10.0
    unfixable_cap = 10.0

    for name, r in named_results:
        if not r.passed:
            all_passed = False
            VALIDATOR_FAILURES.labels(validator=name).inc()
        failures.extend(r.failures)
        score_cap = min(score_cap, r.score_cap)
        unfixable_cap = min(unfixable_cap, r.unfixable_cap)

    return all_passed, failures, score_cap, unfixable_cap

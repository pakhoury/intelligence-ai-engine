"""
Metric compiler — converts structured MetricSQL definitions into
executable Oracle SQL. Fully deterministic: same metric + same params = same SQL.

Two compilation modes:
  1. Template-based: for complex multi-table metrics, substitutes params into
     a pre-written SQL template.
  2. Clause-based: for standard metrics, assembles SELECT/FROM/WHERE/GROUP BY/
     ORDER BY from structured definitions.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .registry import MetricDefinition, MetricSQL


@dataclass(frozen=True)
class CompiledSQL:
    """Output of the metric compiler — deterministic SQL ready for execution."""
    metric_id: str
    version: str
    sql: str
    parameters_used: dict[str, str]
    compilation_mode: str


_DANGEROUS_SQL_RE = re.compile(r"(--|/\*|\*/|;)")


def _sanitize_param(value: str, param_type: str = "string") -> str:
    """Sanitize a parameter value to prevent SQL injection."""
    if param_type == "integer":
        cleaned = re.sub(r"[^\d-]", "", value)
        return cleaned if cleaned and cleaned != "-" else "0"
    sanitized = value.replace("'", "''")
    sanitized = _DANGEROUS_SQL_RE.sub("", sanitized)
    return sanitized


def _substitute_params(text: str, params: dict[str, str]) -> str:
    """Replace {param_name} placeholders with actual values."""
    result = text
    for key, value in params.items():
        result = result.replace(f"{{{key}}}", value)
    return result


def _has_unresolved_params(text: str) -> bool:
    return bool(re.search(r"\{[a-z_]+\}", text))


def _compile_from_template(
    metric: MetricDefinition, params: dict[str, str],
) -> str:
    assert metric.sql is not None and metric.sql.template is not None
    return _substitute_params(metric.sql.template.strip(), params)


def _compile_from_clauses(
    metric: MetricDefinition, params: dict[str, str],
) -> str:
    sql_def: MetricSQL = metric.sql  # type: ignore[assignment]

    select_parts = [
        f"{s.expr} AS {s.alias}" for s in sql_def.select
    ]
    select_clause = ",\n    ".join(select_parts)

    from_clause = sql_def.base_table

    where_parts = []
    for f in sql_def.filters:
        if f.required:
            if f.op == "IS NOT" and f.value == "NULL":
                where_parts.append(f"{f.column} IS NOT NULL")
            else:
                where_parts.append(f"{f.column} {f.op} {f.value}")
            continue

        if f.optional and f.param and f.param not in params:
            continue

        if f.param:
            resolved_value = _substitute_params(f.value, params)
            if _has_unresolved_params(resolved_value):
                continue
            where_parts.append(f"{f.column} {f.op} {resolved_value}")
        else:
            where_parts.append(f"{f.column} {f.op} {f.value}")

    where_clause = ""
    if where_parts:
        where_clause = "WHERE " + "\n  AND ".join(where_parts)

    group_by_clause = ""
    if sql_def.group_by:
        group_by_clause = "GROUP BY " + ", ".join(sql_def.group_by)

    order_by_clause = ""
    if sql_def.order_by:
        order_by_parts = [
            f"{o.expr} {o.direction}" for o in sql_def.order_by
        ]
        order_by_clause = "ORDER BY " + ", ".join(order_by_parts)

    parts = [f"SELECT\n    {select_clause}", f"FROM {from_clause}"]
    if where_clause:
        parts.append(where_clause)
    if group_by_clause:
        parts.append(group_by_clause)
    if order_by_clause:
        parts.append(order_by_clause)
    parts.append("FETCH FIRST 500 ROWS ONLY")

    return "\n".join(parts)


class MetricCompiler:
    """Compiles MetricDefinition → executable SQL."""

    def compile(
        self,
        metric: MetricDefinition,
        params: dict[str, str] | None = None,
    ) -> CompiledSQL | None:
        if not metric.is_compilable:
            return None

        effective_params = metric.get_default_params()
        if params:
            effective_params.update(params)

        param_types = {p.name: p.type for p in metric.parameters}
        effective_params = {
            k: _sanitize_param(v, param_types.get(k, "string"))
            for k, v in effective_params.items()
        }

        if metric.sql.template:  # type: ignore[union-attr]
            sql = _compile_from_template(metric, effective_params)
            mode = "template"
        else:
            sql = _compile_from_clauses(metric, effective_params)
            mode = "clause"

        return CompiledSQL(
            metric_id=metric.metric_id,
            version=metric.version,
            sql=sql,
            parameters_used=effective_params,
            compilation_mode=mode,
        )

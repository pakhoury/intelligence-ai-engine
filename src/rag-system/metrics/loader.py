from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

from .registry import (
    MetricDefinition,
    MetricParameter,
    MetricRegistry,
    MetricSQL,
    SQLFilter,
    SQLOrderBy,
    SQLSelectExpr,
)

DEFAULT_CATALOG_PATH = Path(__file__).parent.parent / "metrics_catalog.yaml"


def _parse_sql(raw: dict) -> MetricSQL:
    select = tuple(
        SQLSelectExpr(expr=s["expr"], alias=s["alias"])
        for s in raw.get("select", [])
    )
    filters = tuple(
        SQLFilter(
            column=f["column"],
            op=f["op"],
            value=f["value"],
            param=f.get("param"),
            optional=f.get("optional", False),
            required=f.get("required", False),
        )
        for f in raw.get("filters", [])
    )
    group_by = tuple(raw.get("group_by", []))
    order_by = tuple(
        SQLOrderBy(expr=o["expr"], direction=o.get("direction", "ASC"))
        for o in raw.get("order_by", [])
    )
    return MetricSQL(
        base_table=raw.get("base_table"),
        select=select,
        filters=filters,
        group_by=group_by,
        order_by=order_by,
        template=raw.get("template"),
    )


def _parse_metric(metric_id: str, raw: dict) -> MetricDefinition:
    parameters = tuple(
        MetricParameter(
            name=p["name"],
            description=p["description"],
            type=p.get("type", "string"),
            default=str(p["default"]) if p.get("default") is not None else None,
            optional=p.get("optional", False),
        )
        for p in raw.get("parameters", [])
    )

    sql = _parse_sql(raw["sql"]) if "sql" in raw else None

    return MetricDefinition(
        metric_id=metric_id,
        name=raw["name"],
        version=raw.get("version", "1.0"),
        description=raw.get("description", "").strip(),
        category=raw.get("category", ""),
        unit=raw.get("unit", ""),
        formula=raw.get("formula", ""),
        keywords=tuple(raw.get("keywords", [])),
        tables=tuple(raw.get("tables", [])),
        sql=sql,
        parameters=parameters,
        steps=tuple(raw.get("steps", [])),
        thresholds=raw.get("thresholds", {}),
        interpretation=raw.get("interpretation", "").strip(),
    )


def load_metrics_catalog(
    path: Union[str, Path, None] = None,
) -> MetricRegistry:
    catalog_path = Path(path) if path else DEFAULT_CATALOG_PATH

    with open(catalog_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    registry = MetricRegistry()
    for metric_id, raw in data.get("metrics", {}).items():
        metric = _parse_metric(metric_id, raw)
        registry.register(metric)

    return registry

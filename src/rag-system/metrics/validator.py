"""
Validates metric definitions against the database catalog,
catching misconfigurations before they reach runtime.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from .registry import MetricDefinition, MetricRegistry

CATALOG_DIR = Path(__file__).parent.parent / "catalog"
LEGACY_CATALOG_PATH = Path(__file__).parent.parent / "catalog.json"


def _load_catalog_columns(catalog_path: Path = CATALOG_DIR) -> Dict[str, Set[str]]:
    """Load table→columns mapping from the database catalog.

    Supports both the new per-table catalog directory structure
    and the legacy single-file catalog.json.
    """
    table_columns: Dict[str, Set[str]] = {}

    if catalog_path.is_dir():
        meta_file = catalog_path / "database_metadata.json"
        if not meta_file.exists():
            return table_columns
        with open(meta_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        for schema_data in metadata.get("schemas", {}).values():
            for table_name, entry in schema_data.get("tables", {}).items():
                table_file = catalog_path / entry.get("file", "")
                if table_file.exists():
                    with open(table_file, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    table_columns[table_name] = set(data.get("columns", {}).keys())
        return table_columns

    resolved = catalog_path if catalog_path.is_file() else LEGACY_CATALOG_PATH
    if not resolved.exists():
        return table_columns
    with open(resolved, "r", encoding="utf-8") as f:
        catalog = json.load(f)
    for schema_data in catalog.get("schemas", {}).values():
        for table_name, info in schema_data.get("tables", {}).items():
            table_columns[table_name] = set(info.get("columns", {}).keys())
    return table_columns


def validate_metric(
    metric: MetricDefinition,
    catalog_path: Path = CATALOG_DIR,
) -> Tuple[bool, List[str]]:
    """Validate a single metric definition.

    Returns (is_valid, list_of_errors).
    """
    errors: List[str] = []

    if not metric.name:
        errors.append(f"{metric.metric_id}: missing name")
    if not metric.version:
        errors.append(f"{metric.metric_id}: missing version")
    if not metric.formula:
        errors.append(f"{metric.metric_id}: missing formula")
    if not metric.tables:
        errors.append(f"{metric.metric_id}: no tables declared")
    if not metric.keywords:
        errors.append(f"{metric.metric_id}: no keywords — metric cannot be resolved from questions")

    table_columns = _load_catalog_columns(catalog_path)
    for table in metric.tables:
        if table not in table_columns:
            errors.append(
                f"{metric.metric_id}: table '{table}' not found in catalog. "
                f"Available: {sorted(table_columns.keys())}"
            )

    if metric.sql and metric.sql.base_table:
        if metric.sql.base_table not in table_columns:
            errors.append(
                f"{metric.metric_id}: sql.base_table '{metric.sql.base_table}' "
                f"not in catalog"
            )
        else:
            valid_cols = table_columns[metric.sql.base_table]
            for filt in metric.sql.filters:
                if filt.column not in valid_cols:
                    errors.append(
                        f"{metric.metric_id}: filter column '{filt.column}' "
                        f"not in {metric.sql.base_table}. "
                        f"Valid: {sorted(valid_cols)}"
                    )

    param_names = {p.name for p in metric.parameters}
    if metric.sql:
        for filt in metric.sql.filters:
            if filt.param and not filt.optional and not filt.required:
                if filt.param not in param_names:
                    errors.append(
                        f"{metric.metric_id}: filter references param '{filt.param}' "
                        f"but it's not declared in parameters"
                    )

    return len(errors) == 0, errors


def validate_catalog(
    registry: MetricRegistry,
    catalog_path: Path = CATALOG_DIR,
) -> Tuple[bool, List[str]]:
    """Validate all metrics in a registry."""
    all_errors: List[str] = []
    for metric in registry.all_metrics:
        _, errors = validate_metric(metric, catalog_path)
        all_errors.extend(errors)
    return len(all_errors) == 0, all_errors

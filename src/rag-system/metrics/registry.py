from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class SQLSelectExpr:
    expr: str
    alias: str


@dataclass(frozen=True)
class SQLFilter:
    column: str
    op: str
    value: str
    param: Optional[str] = None
    optional: bool = False
    required: bool = False


@dataclass(frozen=True)
class SQLOrderBy:
    expr: str
    direction: str = "ASC"


@dataclass(frozen=True)
class MetricSQL:
    """Structured SQL definition that the compiler converts deterministically."""
    base_table: Optional[str] = None
    select: tuple[SQLSelectExpr, ...] = ()
    filters: tuple[SQLFilter, ...] = ()
    group_by: tuple[str, ...] = ()
    order_by: tuple[SQLOrderBy, ...] = ()
    template: Optional[str] = None


@dataclass(frozen=True)
class MetricParameter:
    name: str
    description: str
    type: str = "string"
    default: Optional[str] = None
    optional: bool = False


@dataclass(frozen=True)
class MetricDefinition:
    """Immutable, versioned metric definition — the unit of the registry."""
    metric_id: str
    name: str
    version: str
    description: str
    category: str
    unit: str
    formula: str
    keywords: tuple[str, ...] = ()
    tables: tuple[str, ...] = ()
    sql: Optional[MetricSQL] = None
    parameters: tuple[MetricParameter, ...] = ()
    steps: tuple[dict, ...] = ()
    thresholds: dict = field(default_factory=dict)
    interpretation: str = ""

    @property
    def qualified_id(self) -> str:
        return f"{self.metric_id}@{self.version}"

    @property
    def is_compilable(self) -> bool:
        if self.sql is None:
            return False
        return self.sql.template is not None or self.sql.base_table is not None

    def get_default_params(self) -> Dict[str, str]:
        return {
            p.name: str(p.default)
            for p in self.parameters
            if p.default is not None
        }


class MetricRegistry:
    """Thread-safe, versioned store of metric definitions."""

    def __init__(self):
        self._metrics: Dict[str, MetricDefinition] = {}
        self._by_version: Dict[str, MetricDefinition] = {}

    def register(self, metric: MetricDefinition) -> None:
        self._metrics[metric.metric_id] = metric
        self._by_version[metric.qualified_id] = metric

    def get(self, metric_id: str, version: Optional[str] = None) -> Optional[MetricDefinition]:
        if version:
            return self._by_version.get(f"{metric_id}@{version}")
        return self._metrics.get(metric_id)

    @property
    def metric_ids(self) -> List[str]:
        return list(self._metrics.keys())

    @property
    def all_metrics(self) -> List[MetricDefinition]:
        return list(self._metrics.values())

    def __len__(self) -> int:
        return len(self._metrics)

    def __contains__(self, metric_id: str) -> bool:
        return metric_id in self._metrics

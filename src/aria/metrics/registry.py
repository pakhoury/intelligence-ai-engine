from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SQLSelectExpr:
    expr: str
    alias: str


@dataclass(frozen=True)
class SQLFilter:
    column: str
    op: str
    value: str
    param: str | None = None
    optional: bool = False
    required: bool = False


@dataclass(frozen=True)
class SQLOrderBy:
    expr: str
    direction: str = "ASC"


@dataclass(frozen=True)
class MetricSQL:
    """Structured SQL definition that the compiler converts deterministically."""
    base_table: str | None = None
    select: tuple[SQLSelectExpr, ...] = ()
    filters: tuple[SQLFilter, ...] = ()
    group_by: tuple[str, ...] = ()
    order_by: tuple[SQLOrderBy, ...] = ()
    template: str | None = None


@dataclass(frozen=True)
class MetricParameter:
    name: str
    description: str
    type: str = "string"
    default: str | None = None
    optional: bool = False


@dataclass(frozen=True)
class MetricDefinition:
    """Immutable, versioned metric definition — the unit of the registry.

    Governance fields make the spec an accountable artifact: `owner` is the
    team answerable for the calculation, `approval` references the review
    that authorized it (e.g. an MRM ticket), and `rounding` names the
    output rounding policy. The loader refuses catalogs whose metrics lack
    owner or approval — an unapproved spec never enters the registry.
    """
    metric_id: str
    name: str
    version: str
    description: str
    category: str
    unit: str
    formula: str
    owner: str = ""
    approval: str = ""
    rounding: str = ""
    keywords: tuple[str, ...] = ()
    tables: tuple[str, ...] = ()
    sql: MetricSQL | None = None
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

    def get_default_params(self) -> dict[str, str]:
        return {
            p.name: str(p.default)
            for p in self.parameters
            if p.default is not None
        }


class MetricRegistry:
    """Thread-safe, versioned store of metric definitions."""

    def __init__(self):
        self._metrics: dict[str, MetricDefinition] = {}
        self._by_version: dict[str, MetricDefinition] = {}

    def register(self, metric: MetricDefinition) -> None:
        self._metrics[metric.metric_id] = metric
        self._by_version[metric.qualified_id] = metric

    def get(self, metric_id: str, version: str | None = None) -> MetricDefinition | None:
        if version:
            return self._by_version.get(f"{metric_id}@{version}")
        return self._metrics.get(metric_id)

    @property
    def metric_ids(self) -> list[str]:
        return list(self._metrics.keys())

    @property
    def all_metrics(self) -> list[MetricDefinition]:
        return list(self._metrics.values())

    def __len__(self) -> int:
        return len(self._metrics)

    def __contains__(self, metric_id: str) -> bool:
        return metric_id in self._metrics

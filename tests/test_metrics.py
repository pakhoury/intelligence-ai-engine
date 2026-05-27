"""
Tests for the metrics package: registry, loader, resolver, compiler, validator.

Also includes legacy MetricsCatalog tests and workflow integration tests
to verify the metric governance layer is deterministic:
same question + same metric definition = same SQL, every time.
"""
import os
import sys
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "rag-system"))

from metrics.registry import (
    MetricDefinition,
    MetricParameter,
    MetricRegistry,
    MetricSQL,
    SQLFilter,
    SQLOrderBy,
    SQLSelectExpr,
)
from metrics.loader import load_metrics_catalog
from metrics.resolver import MetricResolver, ResolvedMetric
from metrics.compiler import MetricCompiler, CompiledSQL, _sanitize_param


# ═════════════════════════════════════════════════════════════════════════
# Registry
# ═════════════════════════════════════════════════════════════════════════

class TestMetricRegistry:
    def test_register_and_get(self):
        registry = MetricRegistry()
        metric = MetricDefinition(
            metric_id="test_metric",
            name="Test Metric",
            version="1.0",
            description="A test",
            category="test",
            unit="%",
            formula="count / total * 100",
        )
        registry.register(metric)

        assert "test_metric" in registry
        assert registry.get("test_metric") is metric
        assert registry.get("test_metric", version="1.0") is metric
        assert registry.get("test_metric", version="2.0") is None
        assert len(registry) == 1

    def test_versioned_lookup(self):
        registry = MetricRegistry()
        v1 = MetricDefinition(
            metric_id="m", name="M", version="1.0",
            description="", category="", unit="", formula="old",
        )
        v2 = MetricDefinition(
            metric_id="m", name="M", version="2.0",
            description="", category="", unit="", formula="new",
        )
        registry.register(v1)
        registry.register(v2)

        assert registry.get("m") is v2
        assert registry.get("m", version="1.0") is v1
        assert registry.get("m", version="2.0") is v2

    def test_qualified_id(self):
        metric = MetricDefinition(
            metric_id="abc", name="ABC", version="2.3",
            description="", category="", unit="", formula="",
        )
        assert metric.qualified_id == "abc@2.3"

    def test_is_compilable_no_sql(self):
        metric = MetricDefinition(
            metric_id="a", name="A", version="1.0",
            description="", category="", unit="", formula="",
        )
        assert not metric.is_compilable

    def test_is_compilable_with_base_table(self):
        metric = MetricDefinition(
            metric_id="b", name="B", version="1.0",
            description="", category="", unit="", formula="",
            sql=MetricSQL(base_table="FOO"),
        )
        assert metric.is_compilable

    def test_is_compilable_with_template(self):
        metric = MetricDefinition(
            metric_id="c", name="C", version="1.0",
            description="", category="", unit="", formula="",
            sql=MetricSQL(template="SELECT 1"),
        )
        assert metric.is_compilable

    def test_default_params(self):
        metric = MetricDefinition(
            metric_id="c", name="C", version="1.0",
            description="", category="", unit="", formula="",
            parameters=(
                MetricParameter(name="lookback_months", description="", default="12"),
                MetricParameter(name="department", description="", optional=True),
            ),
        )
        defaults = metric.get_default_params()
        assert defaults == {"lookback_months": "12"}

    def test_metric_ids_list(self):
        registry = MetricRegistry()
        for i in range(3):
            registry.register(MetricDefinition(
                metric_id=f"m{i}", name=f"M{i}", version="1.0",
                description="", category="", unit="", formula="",
            ))
        assert set(registry.metric_ids) == {"m0", "m1", "m2"}


# ═════════════════════════════════════════════════════════════════════════
# Loader
# ═════════════════════════════════════════════════════════════════════════

class TestLoader:
    def test_load_catalog(self):
        registry = load_metrics_catalog()
        assert len(registry) == 7
        assert "compliance_effectiveness_score" in registry
        assert "department_risk_score" in registry

    def test_loaded_metric_has_sql(self):
        registry = load_metrics_catalog()
        m = registry.get("compliance_effectiveness_score")
        assert m is not None
        assert m.sql is not None
        assert m.sql.base_table == "COMPLIANCE_VIOLATIONS"
        assert len(m.sql.select) == 3
        assert m.version == "1.0"

    def test_loaded_metric_has_keywords(self):
        registry = load_metrics_catalog()
        m = registry.get("compliance_effectiveness_score")
        assert len(m.keywords) > 0
        assert "compliance effectiveness" in m.keywords

    def test_template_metric(self):
        registry = load_metrics_catalog()
        m = registry.get("department_risk_score")
        assert m.sql is not None
        assert m.sql.template is not None
        assert m.sql.base_table is None

    def test_all_metrics_have_versions(self):
        registry = load_metrics_catalog()
        for m in registry.all_metrics:
            assert m.version, f"{m.metric_id} missing version"

    def test_all_metrics_have_keywords(self):
        registry = load_metrics_catalog()
        for m in registry.all_metrics:
            assert len(m.keywords) > 0, f"{m.metric_id} has no keywords"

    def test_all_metrics_are_compilable(self):
        registry = load_metrics_catalog()
        for m in registry.all_metrics:
            assert m.is_compilable, f"{m.metric_id} is not compilable"


# ═════════════════════════════════════════════════════════════════════════
# Resolver
# ═════════════════════════════════════════════════════════════════════════

class TestResolver:
    @pytest.fixture
    def resolver(self):
        return MetricResolver(load_metrics_catalog())

    def test_exact_match(self, resolver):
        result = resolver.resolve("What is the compliance effectiveness score?")
        assert result is not None
        assert result.metric.metric_id == "compliance_effectiveness_score"
        assert result.confidence == 1.0

    def test_partial_match(self, resolver):
        result = resolver.resolve("Show me violation severity breakdown")
        assert result is not None
        assert result.metric.metric_id == "violation_severity_distribution"

    def test_no_match(self, resolver):
        result = resolver.resolve("What is the weather like today?")
        assert result is None

    def test_mttr_alias(self, resolver):
        result = resolver.resolve("What is the MTTR for critical violations?")
        assert result is not None
        assert result.metric.metric_id == "mean_time_to_resolution"

    def test_parameter_extraction_months(self, resolver):
        result = resolver.resolve("compliance effectiveness score for the last 6 months")
        assert result is not None
        assert result.extracted_params.get("lookback_months") == "6"

    def test_parameter_extraction_severity(self, resolver):
        result = resolver.resolve("mean time to resolution for critical violations")
        assert result is not None
        assert result.extracted_params.get("severity") == "Critical"

    def test_deterministic_across_calls(self, resolver):
        """Same question must resolve to same metric every time."""
        question = "What is the compliance effectiveness score?"
        results = [resolver.resolve(question) for _ in range(10)]
        metric_ids = {r.metric.metric_id for r in results}
        confidences = {r.confidence for r in results}
        assert len(metric_ids) == 1
        assert len(confidences) == 1

    def test_regulatory_exposure(self, resolver):
        result = resolver.resolve("What is our regulatory exposure?")
        assert result is not None
        assert result.metric.metric_id == "regulatory_exposure_index"

    def test_audit_resolution(self, resolver):
        result = resolver.resolve("What is the audit finding resolution rate?")
        assert result is not None
        assert result.metric.metric_id == "audit_finding_resolution_rate"

    def test_control_coverage(self, resolver):
        result = resolver.resolve("Show control coverage ratio")
        assert result is not None
        assert result.metric.metric_id == "control_coverage_ratio"

    def test_department_risk(self, resolver):
        result = resolver.resolve("Which department has the highest risk score?")
        assert result is not None
        assert result.metric.metric_id == "department_risk_score"

    def test_empty_question(self, resolver):
        assert resolver.resolve("") is None
        assert resolver.resolve("   ") is None

    def test_stopword_only_question(self, resolver):
        assert resolver.resolve("what is the") is None


# ═════════════════════════════════════════════════════════════════════════
# Param sanitization
# ═════════════════════════════════════════════════════════════════════════

class TestSanitizeParam:
    def test_string_escapes_single_quotes(self):
        assert _sanitize_param("O'Brien", "string") == "O''Brien"

    def test_string_strips_comments(self):
        assert "--" not in _sanitize_param("value -- comment", "string")
        assert "/*" not in _sanitize_param("value /* block */", "string")

    def test_string_strips_semicolons(self):
        assert ";" not in _sanitize_param("value; DROP", "string")

    def test_integer_strips_non_numeric(self):
        assert _sanitize_param("12; DROP TABLE", "integer") == "12"

    def test_integer_preserves_negative(self):
        assert _sanitize_param("-6", "integer") == "-6"

    def test_integer_garbage_returns_zero(self):
        assert _sanitize_param("abc", "integer") == "0"

    def test_integer_empty_returns_zero(self):
        assert _sanitize_param("", "integer") == "0"

    def test_clean_string_unchanged(self):
        assert _sanitize_param("Legal", "string") == "Legal"

    def test_clean_integer_unchanged(self):
        assert _sanitize_param("12", "integer") == "12"


# ═════════════════════════════════════════════════════════════════════════
# Compiler
# ═════════════════════════════════════════════════════════════════════════

class TestCompiler:
    @pytest.fixture
    def compiler(self):
        return MetricCompiler()

    @pytest.fixture
    def registry(self):
        return load_metrics_catalog()

    def test_compile_clause_based(self, compiler, registry):
        metric = registry.get("compliance_effectiveness_score")
        result = compiler.compile(metric)
        assert result is not None
        assert result.compilation_mode == "clause"
        assert "SELECT" in result.sql
        assert "COMPLIANCE_VIOLATIONS" in result.sql
        assert "FETCH FIRST 500 ROWS ONLY" in result.sql
        assert "NULLIF" in result.sql

    def test_compile_template_based(self, compiler, registry):
        metric = registry.get("department_risk_score")
        result = compiler.compile(metric)
        assert result is not None
        assert result.compilation_mode == "template"
        assert "composite_risk_score" in result.sql
        assert "COMPLIANCE_VIOLATIONS" in result.sql
        assert "AUDIT_FINDINGS" in result.sql
        assert "RISK_EVENTS" in result.sql

    def test_compile_with_params(self, compiler, registry):
        metric = registry.get("compliance_effectiveness_score")
        result = compiler.compile(metric, {"lookback_months": "6"})
        assert "ADD_MONTHS(SYSDATE, -6)" in result.sql

    def test_compile_default_params_applied(self, compiler, registry):
        metric = registry.get("compliance_effectiveness_score")
        result = compiler.compile(metric)
        assert "ADD_MONTHS(SYSDATE, -12)" in result.sql

    def test_compile_with_optional_param(self, compiler, registry):
        metric = registry.get("compliance_effectiveness_score")

        without_dept = compiler.compile(metric)
        assert "DEPARTMENT" not in without_dept.sql

        with_dept = compiler.compile(metric, {"department": "Operations"})
        assert "DEPARTMENT = 'Operations'" in with_dept.sql

    def test_compile_deterministic(self, compiler, registry):
        """Same metric + same params = byte-identical SQL, every time."""
        metric = registry.get("compliance_effectiveness_score")
        params = {"lookback_months": "6", "department": "IT"}

        results = [compiler.compile(metric, params).sql for _ in range(20)]
        assert len(set(results)) == 1, "Compiler produced different SQL across runs"

    def test_compile_severity_distribution_ordering(self, compiler, registry):
        metric = registry.get("violation_severity_distribution")
        result = compiler.compile(metric)
        assert "ORDER BY" in result.sql
        assert "Critical" in result.sql

    def test_compile_mean_time_to_resolution(self, compiler, registry):
        metric = registry.get("mean_time_to_resolution")
        result = compiler.compile(metric)
        assert "STATUS = 'Closed'" in result.sql
        assert "CLOSURE_DATE IS NOT NULL" in result.sql
        assert "AVG" in result.sql

    def test_compile_regulatory_exposure_required_filter(self, compiler, registry):
        metric = registry.get("regulatory_exposure_index")
        result = compiler.compile(metric)
        assert "STATUS != 'Closed'" in result.sql
        assert "ORDER BY" in result.sql

    def test_sql_injection_single_quote_escaped(self, compiler, registry):
        metric = registry.get("compliance_effectiveness_score")
        result = compiler.compile(metric, {"department": "Legal' OR 1=1 --"})
        assert "Legal''" in result.sql
        assert "' OR" not in result.sql.replace("''", "")

    def test_sql_injection_semicolon_stripped(self, compiler, registry):
        metric = registry.get("compliance_effectiveness_score")
        result = compiler.compile(metric, {"department": "Legal; DROP TABLE users"})
        assert ";" not in result.sql or result.sql.count(";") == 0
        assert "DROP" not in result.sql.upper().split("'")[0]

    def test_sql_injection_comment_stripped(self, compiler, registry):
        metric = registry.get("compliance_effectiveness_score")
        result = compiler.compile(metric, {"department": "Legal -- comment"})
        assert "--" not in result.sql

    def test_sql_injection_integer_param_sanitized(self, compiler, registry):
        metric = registry.get("compliance_effectiveness_score")
        result = compiler.compile(metric, {"lookback_months": "12; DROP TABLE users"})
        assert "12" in result.sql
        assert "DROP" not in result.sql

    def test_sql_injection_block_comment_stripped(self, compiler, registry):
        metric = registry.get("compliance_effectiveness_score")
        result = compiler.compile(metric, {"department": "Legal /* injected */"})
        assert "/*" not in result.sql
        assert "*/" not in result.sql

    def test_non_compilable_metric(self, compiler):
        metric = MetricDefinition(
            metric_id="x", name="X", version="1.0",
            description="", category="", unit="", formula="",
        )
        assert compiler.compile(metric) is None

    def test_compiled_sql_contains_metric_id_and_version(self, compiler, registry):
        metric = registry.get("compliance_effectiveness_score")
        result = compiler.compile(metric)
        assert result.metric_id == "compliance_effectiveness_score"
        assert result.version == "1.0"


# ═════════════════════════════════════════════════════════════════════════
# End-to-end determinism
# ═════════════════════════════════════════════════════════════════════════

class TestEndToEndDeterminism:
    """Prove that question -> SQL is fully deterministic through the registry path."""

    QUESTIONS_AND_EXPECTED = [
        (
            "What is the compliance effectiveness score?",
            "compliance_effectiveness_score",
        ),
        (
            "Show violation severity breakdown",
            "violation_severity_distribution",
        ),
        (
            "What is the audit finding resolution rate?",
            "audit_finding_resolution_rate",
        ),
        (
            "What is our regulatory exposure?",
            "regulatory_exposure_index",
        ),
        (
            "Which department has the highest risk score?",
            "department_risk_score",
        ),
        (
            "Show control coverage ratio",
            "control_coverage_ratio",
        ),
        (
            "What is the MTTR?",
            "mean_time_to_resolution",
        ),
    ]

    def test_all_metrics_resolve_deterministically(self):
        registry = load_metrics_catalog()
        resolver = MetricResolver(registry)
        compiler = MetricCompiler()

        for question, expected_metric_id in self.QUESTIONS_AND_EXPECTED:
            resolved = resolver.resolve(question)
            assert resolved is not None, f"Failed to resolve: {question}"
            assert resolved.metric.metric_id == expected_metric_id, (
                f"'{question}' resolved to '{resolved.metric.metric_id}' "
                f"instead of '{expected_metric_id}'"
            )

            compiled = compiler.compile(resolved.metric, resolved.extracted_params)
            assert compiled is not None, f"Failed to compile: {expected_metric_id}"

            compiled_again = compiler.compile(resolved.metric, resolved.extracted_params)
            assert compiled.sql == compiled_again.sql, (
                f"Non-deterministic SQL for {expected_metric_id}"
            )

    def test_sql_identical_across_independent_pipelines(self):
        """Two independent registry+resolver+compiler chains produce identical SQL."""
        question = "What is the compliance effectiveness score for the last 6 months?"

        registry_a = load_metrics_catalog()
        resolver_a = MetricResolver(registry_a)
        compiler_a = MetricCompiler()

        registry_b = load_metrics_catalog()
        resolver_b = MetricResolver(registry_b)
        compiler_b = MetricCompiler()

        resolved_a = resolver_a.resolve(question)
        resolved_b = resolver_b.resolve(question)
        assert resolved_a.metric.metric_id == resolved_b.metric.metric_id

        compiled_a = compiler_a.compile(resolved_a.metric, resolved_a.extracted_params)
        compiled_b = compiler_b.compile(resolved_b.metric, resolved_b.extracted_params)
        assert compiled_a.sql == compiled_b.sql


# ═════════════════════════════════════════════════════════════════════════
# Legacy MetricsCatalog tests (backward compatibility)
# ═════════════════════════════════════════════════════════════════════════

class TestMetricsCatalog:
    def test_loads_all_metrics(self):
        from metrics import MetricsCatalog
        catalog = MetricsCatalog()
        ids = catalog.metric_ids
        assert len(ids) >= 7
        assert "compliance_effectiveness_score" in ids

    def test_get_returns_metric(self):
        from metrics import MetricsCatalog
        catalog = MetricsCatalog()
        m = catalog.get("compliance_effectiveness_score")
        assert m is not None
        assert m["name"] == "Compliance Effectiveness Score"
        assert m["unit"] == "%"
        assert "COMPLIANCE_VIOLATIONS" in m["tables"]

    def test_get_unknown_returns_none(self):
        from metrics import MetricsCatalog
        catalog = MetricsCatalog()
        assert catalog.get("nonexistent_metric") is None

    def test_reload(self):
        from metrics import MetricsCatalog
        catalog = MetricsCatalog()
        count_before = len(catalog.metric_ids)
        catalog.reload()
        assert len(catalog.metric_ids) == count_before


class TestBuildAllContext:
    def test_includes_all_metrics(self):
        from metrics import MetricsCatalog
        ctx = MetricsCatalog().build_all_context()
        assert "Compliance Effectiveness Score" in ctx
        assert "Department Risk Score" in ctx
        assert "Mean Time to Resolution" in ctx

    def test_includes_formula_and_steps(self):
        from metrics import MetricsCatalog
        ctx = MetricsCatalog().build_all_context()
        assert "Formula:" in ctx
        assert "Steps:" in ctx
        assert "Step 1:" in ctx

    def test_includes_parameters(self):
        from metrics import MetricsCatalog
        ctx = MetricsCatalog().build_all_context()
        assert "Parameters:" in ctx
        assert "lookback_months" in ctx

    def test_includes_tables(self):
        from metrics import MetricsCatalog
        ctx = MetricsCatalog().build_all_context()
        assert "COMPLIANCE_VIOLATIONS" in ctx
        assert "AUDIT_FINDINGS" in ctx

    def test_empty_catalog_returns_empty(self):
        from metrics import MetricsCatalog
        import tempfile
        path = os.path.join(tempfile.gettempdir(), "empty_metrics.yaml")
        with open(path, "w") as f:
            f.write("metrics: {}")
        catalog = MetricsCatalog(path)
        assert catalog.build_all_context() == ""
        os.remove(path)


# ═════════════════════════════════════════════════════════════════════════
# Workflow integration: resolved metric context flows to LLM SQL
# ═════════════════════════════════════════════════════════════════════════

class TestResolvedMetricFlowsToLlm:
    """When metric resolver matches a metric, its full definition is passed
    to the LLM as context for SQL generation."""

    @patch("nodes.redis_client")
    @patch("nodes._ainvoke_llm", new_callable=AsyncMock)
    @patch("nodes.db_connector")
    @patch("nodes._get_vector_store")
    @patch("nodes._get_metrics_catalog")
    @patch("nodes._get_metric_resolver")
    async def test_resolved_metric_context_in_sql_prompt(
        self, mock_resolver_fn,
        mock_catalog_fn, mock_vs_fn, mock_db,
        mock_ainvoke, mock_redis,
    ):
        from metrics import MetricsCatalog
        mock_catalog_fn.return_value = MetricsCatalog()

        mock_redis.get.return_value = None
        mock_redis.set.return_value = True

        registry = load_metrics_catalog()
        keyword_resolver = MetricResolver(registry)

        resolved = keyword_resolver.resolve("What is the compliance effectiveness score?")
        mock_resolver = MagicMock()
        mock_resolver.resolve = AsyncMock(return_value=resolved)
        mock_resolver_fn.return_value = mock_resolver

        mock_db.build_table_selection_prompt.return_value = "table selection prompt"
        mock_db.parse_table_selection.return_value = ["COMPLIANCE_VIOLATIONS"]
        mock_db.get_relevant_tables.return_value = ["COMPLIANCE_VIOLATIONS"]
        mock_db.get_table_schema.return_value = "Table: COMPLIANCE_VIOLATIONS"
        mock_db.validate_columns.return_value = ""
        mock_db.execute_query.return_value = "[(100, 85, 85.00)]"

        mock_doc = MagicMock()
        mock_doc.metadata = {"title": "Test"}
        mock_doc.page_content = "test"
        mock_store = MagicMock()
        mock_store.similarity_search.return_value = [mock_doc]
        mock_vs_fn.return_value = mock_store

        mock_ainvoke.side_effect = [
            "SQL_ONLY",
            "NO_CLARIFICATION_NEEDED",
            "COMPLIANCE_VIOLATIONS",
            "SELECT COUNT(*) AS total, SUM(CASE WHEN STATUS='Closed' THEN 1 ELSE 0 END) AS closed FROM COMPLIANCE_VIOLATIONS",
            "The compliance effectiveness score is 85%.",
            "Score: 9.0\nDecision: APPROVED",
        ]

        from workflow import app as workflow_app
        result = await workflow_app.ainvoke(
            {
                "question": "What is the compliance effectiveness score?",
                "session_id": "test-metric-flow",
                "conversation_history": [],
            },
            config={"configurable": {"thread_id": "test-metric-llm-flow"}},
        )

        assert result.get("resolved_metric") == "compliance_effectiveness_score"
        assert "compiled_sql" not in result
        assert "85" in result.get("sql_result", "")

        sql_prompt = mock_ainvoke.call_args_list[3][0][0]
        assert "Resolved Metric Definition" in sql_prompt
        assert "(number of closed violations / total number of violations) * 100" in sql_prompt

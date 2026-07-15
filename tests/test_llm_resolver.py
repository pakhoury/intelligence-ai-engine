import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "aria"))

from metrics.llm_resolver import LLMMetricResolver, _build_catalog_context, _parse_response
from metrics.registry import MetricDefinition, MetricParameter, MetricRegistry
from metrics.resolver import ResolvedMetric


def _make_registry():
    registry = MetricRegistry()
    registry.register(MetricDefinition(
        metric_id="compliance_effectiveness_score",
        name="Compliance Effectiveness Score",
        version="1.0",
        description="Percentage of closed violations out of total violations.",
        category="compliance",
        unit="%",
        formula="(closed / total) * 100",
        keywords=("compliance effectiveness",),
        tables=("COMPLIANCE_VIOLATIONS",),
        parameters=(
            MetricParameter(name="lookback_months", description="Months to look back", type="integer", default="12"),
            MetricParameter(name="department", description="Filter by department", type="string", optional=True),
        ),
    ))
    registry.register(MetricDefinition(
        metric_id="violation_severity_distribution",
        name="Violation Severity Distribution",
        version="1.0",
        description="Violations broken down by severity level.",
        category="compliance",
        unit="count",
        formula="Count grouped by severity",
        keywords=("severity distribution",),
        tables=("COMPLIANCE_VIOLATIONS",),
        parameters=(
            MetricParameter(name="lookback_months", description="Months to look back", type="integer", default="12"),
        ),
    ))
    return registry


def _make_resolver(llm_response, confidence_threshold=0.5):
    mock_invoke = AsyncMock(return_value=llm_response)
    mock_load_prompt = MagicMock(return_value="Catalog:\n{metric_catalog}\n\nQuestion: {question}")
    registry = _make_registry()
    resolver = LLMMetricResolver(registry, mock_invoke, mock_load_prompt, confidence_threshold)
    return resolver, mock_invoke, mock_load_prompt


class TestBuildCatalogContext:
    def test_includes_all_metrics(self):
        registry = _make_registry()
        context = _build_catalog_context(registry)
        assert "compliance_effectiveness_score" in context
        assert "violation_severity_distribution" in context

    def test_includes_parameters(self):
        registry = _make_registry()
        context = _build_catalog_context(registry)
        assert "lookback_months" in context
        assert "department" in context
        assert "(optional)" in context


class TestParseResponse:
    def test_valid_json(self):
        raw = json.dumps({"metric_id": "foo", "confidence": 0.9, "reasoning": "match", "extracted_params": {}})
        result = _parse_response(raw)
        assert result["metric_id"] == "foo"

    def test_json_in_markdown_fence(self):
        raw = '```json\n{"metric_id": "foo", "confidence": 0.9}\n```'
        result = _parse_response(raw)
        assert result["metric_id"] == "foo"

    def test_json_in_bare_fence(self):
        raw = '```\n{"metric_id": null}\n```'
        result = _parse_response(raw)
        assert result["metric_id"] is None

    def test_malformed_json(self):
        assert _parse_response("This is not JSON at all") is None

    def test_empty_string(self):
        assert _parse_response("") is None


class TestLLMMetricResolver:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        llm_response = json.dumps({
            "metric_id": "compliance_effectiveness_score",
            "confidence": 0.95,
            "reasoning": "Question asks about compliance effectiveness",
            "extracted_params": {"lookback_months": "6"},
        })
        resolver, mock_invoke, _ = _make_resolver(llm_response)

        result = await resolver.resolve("What is the compliance effectiveness for the last 6 months?")

        assert isinstance(result, ResolvedMetric)
        assert result.metric.metric_id == "compliance_effectiveness_score"
        assert result.confidence == 0.95
        assert result.extracted_params == {"lookback_months": "6"}
        assert result.matched_keyword == "Question asks about compliance effectiveness"
        mock_invoke.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_metric_match(self):
        llm_response = json.dumps({
            "metric_id": None,
            "confidence": 0.1,
            "reasoning": "No metric matches",
            "extracted_params": {},
        })
        resolver, _, _ = _make_resolver(llm_response)

        result = await resolver.resolve("What is the weather today?")
        assert result is None

    @pytest.mark.asyncio
    async def test_low_confidence(self):
        llm_response = json.dumps({
            "metric_id": "compliance_effectiveness_score",
            "confidence": 0.3,
            "reasoning": "Weak match",
            "extracted_params": {},
        })
        resolver, _, _ = _make_resolver(llm_response, confidence_threshold=0.5)

        result = await resolver.resolve("Something vaguely about compliance")
        assert result is None

    @pytest.mark.asyncio
    async def test_unknown_metric_id(self):
        llm_response = json.dumps({
            "metric_id": "nonexistent_metric",
            "confidence": 0.9,
            "reasoning": "Matched",
            "extracted_params": {},
        })
        resolver, _, _ = _make_resolver(llm_response)

        result = await resolver.resolve("Some question")
        assert result is None

    @pytest.mark.asyncio
    async def test_malformed_llm_response(self):
        resolver, _, _ = _make_resolver("I'm sorry, I can't generate JSON right now.")

        result = await resolver.resolve("What is compliance effectiveness?")
        assert result is None

    @pytest.mark.asyncio
    async def test_empty_question(self):
        resolver, mock_invoke, _ = _make_resolver("{}")

        result = await resolver.resolve("")
        assert result is None
        mock_invoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_whitespace_question(self):
        resolver, mock_invoke, _ = _make_resolver("{}")

        result = await resolver.resolve("   ")
        assert result is None
        mock_invoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_arbitrary_params_extracted(self):
        llm_response = json.dumps({
            "metric_id": "compliance_effectiveness_score",
            "confidence": 0.92,
            "reasoning": "User asks about compliance effectiveness for a specific business key",
            "extracted_params": {
                "business_key": "88",
                "person_name": "Arthur Soar",
            },
        })
        resolver, _, _ = _make_resolver(llm_response)

        result = await resolver.resolve(
            "I am Arthur Soar and my business key is 88, what is my compliance effectiveness?"
        )

        assert result is not None
        assert result.extracted_params["business_key"] == "88"
        assert result.extracted_params["person_name"] == "Arthur Soar"

    @pytest.mark.asyncio
    async def test_params_coerced_to_strings(self):
        llm_response = json.dumps({
            "metric_id": "compliance_effectiveness_score",
            "confidence": 0.9,
            "reasoning": "match",
            "extracted_params": {"lookback_months": 6, "department": "Legal"},
        })
        resolver, _, _ = _make_resolver(llm_response)

        result = await resolver.resolve("Compliance effectiveness for Legal last 6 months")

        assert result.extracted_params["lookback_months"] == "6"
        assert result.extracted_params["department"] == "Legal"

    @pytest.mark.asyncio
    async def test_markdown_fenced_response(self):
        inner = json.dumps({
            "metric_id": "violation_severity_distribution",
            "confidence": 0.88,
            "reasoning": "severity breakdown",
            "extracted_params": {},
        })
        llm_response = f"```json\n{inner}\n```"
        resolver, _, _ = _make_resolver(llm_response)

        result = await resolver.resolve("Show me violations by severity")

        assert result is not None
        assert result.metric.metric_id == "violation_severity_distribution"

    @pytest.mark.asyncio
    async def test_prompt_contains_question_and_catalog(self):
        llm_response = json.dumps({
            "metric_id": None,
            "confidence": 0.0,
            "reasoning": "no match",
            "extracted_params": {},
        })
        resolver, mock_invoke, mock_load_prompt = _make_resolver(llm_response)

        await resolver.resolve("What is the compliance effectiveness?")

        prompt_sent = mock_invoke.call_args[0][0]
        assert "What is the compliance effectiveness?" in prompt_sent
        assert "compliance_effectiveness_score" in prompt_sent

    @pytest.mark.asyncio
    async def test_prompt_loader_failure(self):
        mock_invoke = AsyncMock()
        mock_load_prompt = MagicMock(return_value="")
        registry = _make_registry()
        resolver = LLMMetricResolver(registry, mock_invoke, mock_load_prompt)

        result = await resolver.resolve("Some question")
        assert result is None
        mock_invoke.assert_not_called()

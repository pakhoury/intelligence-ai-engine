from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Awaitable, Dict, Optional

from .registry import MetricDefinition, MetricRegistry
from .resolver import ResolvedMetric

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _build_catalog_context(registry: MetricRegistry) -> str:
    sections = []
    for metric in registry.all_metrics:
        lines = [
            f"metric_id: {metric.metric_id}",
            f"  name: {metric.name}",
            f"  description: {metric.description.strip()}",
            f"  category: {metric.category}",
            f"  formula: {metric.formula}",
            f"  tables: {', '.join(metric.tables)}",
        ]
        if metric.parameters:
            lines.append("  parameters:")
            for p in metric.parameters:
                opt = " (optional)" if p.optional else ""
                default = f", default={p.default}" if p.default else ""
                lines.append(
                    f"    - {p.name} ({p.type}{default}{opt}): {p.description}"
                )
        sections.append("\n".join(lines))
    return "\n\n".join(sections)


def _parse_response(content: str) -> Optional[dict]:
    text = content.strip()
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


class LLMMetricResolver:

    def __init__(
        self,
        registry: MetricRegistry,
        llm_invoke_fn: Callable[[str], Awaitable[str]],
        prompt_loader: Callable[[str], str],
        confidence_threshold: float = 0.5,
    ):
        self._registry = registry
        self._invoke = llm_invoke_fn
        self._load_prompt = prompt_loader
        self._threshold = confidence_threshold
        self._catalog_context = _build_catalog_context(registry)

    async def resolve(self, question: str) -> Optional[ResolvedMetric]:
        if not question or not question.strip():
            return None

        prompt_template = self._load_prompt("metric_resolver.txt")
        if not prompt_template:
            return None

        prompt = prompt_template.format(
            question=question,
            metric_catalog=self._catalog_context,
        )

        content = await self._invoke(prompt)
        parsed = _parse_response(content)
        if parsed is None:
            return None

        metric_id = parsed.get("metric_id")
        if metric_id is None:
            return None

        metric = self._registry.get(metric_id)
        if metric is None:
            return None

        confidence = float(parsed.get("confidence", 0.0))
        if confidence < self._threshold:
            return None

        raw_params = parsed.get("extracted_params", {})
        extracted_params = {str(k): str(v) for k, v in raw_params.items()}

        reasoning = parsed.get("reasoning", "")

        return ResolvedMetric(
            metric=metric,
            confidence=confidence,
            matched_keyword=reasoning,
            extracted_params=extracted_params,
        )

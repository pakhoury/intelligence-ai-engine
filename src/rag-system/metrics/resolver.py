"""
Deterministic metric resolver — maps user questions to registered metrics
using keyword matching. No LLM involved.

Resolution flow:
  1. Normalize question (lowercase, collapse whitespace)
  2. Score each metric by keyword overlap
  3. Return best match if score exceeds threshold, else None

This is deliberately conservative: unresolved questions fall back to
the existing LLM-driven SQL path, so false negatives are safe.
False positives (wrong metric) are the real risk — the threshold is set high.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from .registry import MetricDefinition, MetricRegistry

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must",
    "i", "me", "my", "we", "our", "you", "your", "it", "its",
    "this", "that", "these", "those", "what", "which", "who", "whom",
    "how", "when", "where", "why",
    "and", "but", "or", "nor", "not", "no", "so", "if", "then",
    "of", "in", "on", "at", "to", "for", "with", "by", "from", "about",
    "into", "through", "during", "before", "after", "above", "below",
    "up", "down", "out", "off", "over", "under",
    "show", "tell", "give", "get", "find", "list", "display", "calculate",
    "compute", "determine", "provide",
})

_WORD_RE = re.compile(r"[a-z0-9]+")

MATCH_THRESHOLD = 0.45


@dataclass(frozen=True)
class ResolvedMetric:
    metric: MetricDefinition
    confidence: float
    matched_keyword: str
    extracted_params: Dict[str, str]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _tokenize(text: str) -> List[str]:
    return [w for w in _WORD_RE.findall(text) if w not in _STOP_WORDS]


def _keyword_score(question_tokens: List[str], keyword: str) -> float:
    """Score how well a keyword phrase matches the question tokens.

    Uses coverage: what fraction of the keyword's meaningful words
    appear in the question.
    """
    kw_tokens = _tokenize(keyword)
    if not kw_tokens:
        return 0.0
    matched = sum(1 for t in kw_tokens if t in question_tokens)
    return matched / len(kw_tokens)


def _substring_match(question_norm: str, keyword: str) -> bool:
    """Check if the keyword appears as a substring in the question."""
    return keyword.lower() in question_norm


def _extract_params_from_question(
    question: str, metric: MetricDefinition,
) -> Dict[str, str]:
    """Extract parameter values from question text using simple patterns."""
    params: Dict[str, str] = {}
    q_lower = question.lower()

    for p in metric.parameters:
        if p.name == "lookback_months":
            m = re.search(r"(?:last|past)\s+(\d+)\s+months?", q_lower)
            if m:
                params["lookback_months"] = m.group(1)
                continue
            quarter_match = re.search(r"q([1-4])\s+(\d{4})", q_lower)
            if quarter_match:
                params["_quarter"] = quarter_match.group(1)
                params["_year"] = quarter_match.group(2)
                continue
            year_match = re.search(r"\b(20\d{2})\b", q_lower)
            if year_match:
                params["_year"] = year_match.group(1)

        elif p.name == "department":
            dept_match = re.search(
                r"(?:department|dept)[\s:]+([A-Za-z\s&]+?)(?:\s*(?:department|dept|$|\?))",
                question, re.IGNORECASE,
            )
            if dept_match:
                params["department"] = dept_match.group(1).strip()
            for_match = re.search(
                r"for\s+(?:the\s+)?([A-Z][A-Za-z\s&]+?)(?:\s+department|\s*$|\s*\?)",
                question,
            )
            if for_match and "department" not in params:
                params["department"] = for_match.group(1).strip()

        elif p.name == "severity":
            sev_match = re.search(
                r"\b(critical|high|medium|low)\b", q_lower,
            )
            if sev_match:
                params["severity"] = sev_match.group(1).capitalize()

        elif p.name == "audit_type":
            for audit_t in ("Internal Audit", "External Audit", "Regulatory Inspection"):
                if audit_t.lower() in q_lower:
                    params["audit_type"] = audit_t
                    break

    return params


class MetricResolver:
    """Resolves user questions to registered metrics deterministically."""

    def __init__(self, registry: MetricRegistry, threshold: float = MATCH_THRESHOLD):
        self._registry = registry
        self._threshold = threshold

    def resolve(self, question: str) -> Optional[ResolvedMetric]:
        question_norm = _normalize(question)
        question_tokens = _tokenize(question_norm)

        if not question_tokens:
            return None

        best_score = 0.0
        best_keyword = ""
        best_metric: Optional[MetricDefinition] = None

        for metric in self._registry.all_metrics:
            for keyword in metric.keywords:
                if _substring_match(question_norm, keyword):
                    score = 1.0
                else:
                    score = _keyword_score(question_tokens, keyword)

                if score > best_score:
                    best_score = score
                    best_keyword = keyword
                    best_metric = metric

        if best_metric is None or best_score < self._threshold:
            return None

        extracted = _extract_params_from_question(question, best_metric)

        return ResolvedMetric(
            metric=best_metric,
            confidence=best_score,
            matched_keyword=best_keyword,
            extracted_params=extracted,
        )

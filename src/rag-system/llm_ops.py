"""
LLM Operations module — model fallback, cost tracking, context window management.
"""
import os
import time
from typing import Dict, List, Optional
from observability import logger

# ── Model Cost Table (per 1M tokens) ─────────────────────────────────
# Prices in USD. Update as providers change pricing.
MODEL_COSTS: Dict[str, Dict[str, float]] = {
    # Groq (free tier has limits, paid tier pricing)
    "llama-3.3-70b-versatile": {"prompt": 0.59, "completion": 0.79},
    "llama-3.1-8b-instant": {"prompt": 0.05, "completion": 0.08},
    "mixtral-8x7b-32768": {"prompt": 0.24, "completion": 0.24},
    # OpenAI
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    # Anthropic
    "claude-sonnet-4-20250514": {"prompt": 3.00, "completion": 15.00},
    # Google
    "gemini-2.0-flash": {"prompt": 0.10, "completion": 0.40},
}

# ── Context Window Limits ─────────────────────────────────────────────
MODEL_CONTEXT_WINDOWS: Dict[str, int] = {
    "llama-3.3-70b-versatile": 128000,
    "llama-3.1-8b-instant": 131072,
    "mixtral-8x7b-32768": 32768,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "claude-sonnet-4-20250514": 200000,
    "gemini-2.0-flash": 1048576,
}


class CostTracker:
    """Track LLM costs per request and cumulative."""

    def __init__(self, model: str):
        self.model = model
        self.costs = MODEL_COSTS.get(model, {"prompt": 0, "completion": 0})

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost in USD for a single LLM call."""
        prompt_cost = (prompt_tokens / 1_000_000) * self.costs["prompt"]
        completion_cost = (completion_tokens / 1_000_000) * self.costs["completion"]
        return round(prompt_cost + completion_cost, 6)

    def estimate_monthly_cost(
        self, avg_prompt_tokens: int, avg_completion_tokens: int, requests_per_day: int
    ) -> float:
        """Estimate monthly cost based on average usage."""
        cost_per_request = self.calculate_cost(avg_prompt_tokens, avg_completion_tokens)
        return round(cost_per_request * requests_per_day * 30, 2)


class ContextWindowManager:
    """Manage content to fit within model context windows."""

    def __init__(self, model: str, max_output_tokens: int = 1200):
        self.model = model
        self.max_context = MODEL_CONTEXT_WINDOWS.get(model, 128000)
        self.max_output = max_output_tokens
        # Reserve space for output + safety margin
        self.max_input = self.max_context - self.max_output - 500

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimation: ~4 chars per token for English."""
        return len(text) // 4

    def truncate_to_fit(
        self,
        prompt_template: str,
        question: str,
        sql_result: str = "",
        docs: str = "",
        history: str = "",
    ) -> Dict[str, str]:
        """Truncate content to fit within context window, prioritizing by importance.

        Priority: question > sql_result > docs > history
        """
        # Fixed overhead: template + question
        fixed_tokens = self.estimate_tokens(prompt_template) + self.estimate_tokens(question)
        remaining = self.max_input - fixed_tokens

        if remaining <= 0:
            logger.warning(
                "Question alone exceeds context window",
                extra={"node": "context_manager", "error": "question_too_long"},
            )
            return {
                "sql_result": "",
                "docs": "",
                "history": "",
            }

        result = {}

        # Allocate: 40% SQL, 40% docs, 20% history
        sql_budget = int(remaining * 0.4)
        docs_budget = int(remaining * 0.4)
        history_budget = int(remaining * 0.2)

        # SQL result
        sql_tokens = self.estimate_tokens(sql_result)
        if sql_tokens <= sql_budget:
            result["sql_result"] = sql_result
            # Give unused SQL budget to docs
            docs_budget += sql_budget - sql_tokens
        else:
            char_limit = sql_budget * 4
            result["sql_result"] = sql_result[:char_limit] + "\n... [truncated]"

        # Documents
        docs_tokens = self.estimate_tokens(docs)
        if docs_tokens <= docs_budget:
            result["docs"] = docs
            history_budget += docs_budget - docs_tokens
        else:
            char_limit = docs_budget * 4
            result["docs"] = docs[:char_limit] + "\n... [truncated]"

        # History
        history_tokens = self.estimate_tokens(history)
        if history_tokens <= history_budget:
            result["history"] = history
        else:
            char_limit = history_budget * 4
            result["history"] = history[:char_limit] + "\n... [truncated]"

        total_estimated = fixed_tokens + sum(
            self.estimate_tokens(v) for v in result.values()
        )

        if total_estimated > self.max_input:
            logger.warning(
                f"Context may exceed window: ~{total_estimated} tokens (limit {self.max_input})",
                extra={"node": "context_manager"},
            )

        return result


class ModelFallbackChain:
    """Try multiple LLM providers in order, falling back on failure."""

    def __init__(self, models: List[Dict]):
        """
        models: [{"provider": "groq", "model": "llama-3.3-70b-versatile", "create_fn": callable}, ...]
        """
        self.models = models

    def invoke_with_fallback(self, prompt: str) -> Optional[str]:
        """Try each model in sequence until one succeeds."""
        errors = []
        for i, model_config in enumerate(self.models):
            model_name = model_config.get("model", "unknown")
            try:
                llm_instance = model_config["create_fn"]()
                response = llm_instance.invoke(prompt)
                if i > 0:
                    logger.info(
                        f"Fallback to {model_name} succeeded",
                        extra={"node": "fallback", "model": model_name},
                    )
                return response.content
            except Exception as e:
                errors.append(f"{model_name}: {e}")
                logger.warning(
                    f"Model {model_name} failed, trying next",
                    extra={"node": "fallback", "model": model_name, "error": str(e)},
                )

        logger.error(
            f"All models failed: {errors}",
            extra={"node": "fallback", "error": str(errors)},
        )
        return None



PROMPT_VERSIONS: Dict[str, str] = {
    "router.txt": "v2.0-history-aware",
    "clarification.txt": "v1.1-conservative",
    "sql_agent.txt": "v3.0-unified-metric-aware",
    "final_answer.txt": "v2.0-history-aware",
    "reviewer.txt": "v1.0-score-format",
}


def get_prompt_version(filename: str) -> str:
    """Get the version tag for a prompt file."""
    return PROMPT_VERSIONS.get(filename, "unknown")


LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
cost_tracker = CostTracker(LLM_MODEL)
context_manager = ContextWindowManager(LLM_MODEL)

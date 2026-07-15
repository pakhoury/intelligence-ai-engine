import os
from pathlib import Path

import redis
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from database.oracle import OracleConnector
from llm_ops import ModelFallbackChain
from observability import llm_handler

load_dotenv()

# ── LLM Configuration ────────────────────────────────────────────────

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_FALLBACK_MODEL = os.getenv("LLM_FALLBACK_MODEL", "llama-3.1-8b-instant")
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "30"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))


def _create_groq_llm(model: str) -> ChatGroq:
    return ChatGroq(
        model=model,
        temperature=0,
        max_tokens=1200,
        timeout=LLM_TIMEOUT,
        max_retries=LLM_MAX_RETRIES,
        callbacks=[llm_handler],
    )


llm = _create_groq_llm(LLM_MODEL)

# Used when the circuit breaker on the PRIMARY model is open — so the chain
# must start with the fallback model, not the one that is already failing.
FALLBACK_MODELS = [
    {"provider": "groq", "model": LLM_FALLBACK_MODEL, "create_fn": lambda: _create_groq_llm(LLM_FALLBACK_MODEL)},
    {"provider": "groq", "model": LLM_MODEL, "create_fn": lambda: _create_groq_llm(LLM_MODEL)},
]


llm_fallback = ModelFallbackChain(FALLBACK_MODELS)

# ── Database ──────────────────────────────────────────────────────────

db_connector = OracleConnector(llm=llm)

# ── Redis ─────────────────────────────────────────────────────────────

redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    password=os.getenv("REDIS_PASSWORD", None),
    db=0,
    decode_responses=True,
    socket_timeout=5,
)

# ── Prompts ───────────────────────────────────────────────────────────

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(filename: str) -> str:
    try:
        with open(PROMPTS_DIR / filename, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        from observability import logger as _logger
        _logger.error(f"Prompt file not found: {filename}", extra={"error": f"missing_prompt:{filename}"})
        return ""

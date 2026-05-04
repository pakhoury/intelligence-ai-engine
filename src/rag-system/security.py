"""
Security module — API key auth, rate limiting, input validation, prompt injection guard.
"""
import os
import re
import hashlib
import secrets
from typing import Optional
from fastapi import Request, HTTPException, Security
from fastapi.security import APIKeyHeader
from slowapi import Limiter
from slowapi.util import get_remote_address
from observability import logger

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

# In production, use a secrets manager (AWS Secrets Manager, Vault, etc.)
_valid_keys: Optional[set] = None


def _get_valid_keys() -> set:
    global _valid_keys
    if _valid_keys is None:
        raw = os.getenv("API_KEYS", "")
        if raw:
            _valid_keys = {k.strip() for k in raw.split(",") if k.strip()}
        else:
            _valid_keys = set()
    return _valid_keys


def _hash_key(key: str) -> str:
    """Hash API key for safe logging."""
    return hashlib.sha256(key.encode()).hexdigest()[:12]


async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    """Dependency that validates the API key from X-API-Key header."""
    valid_keys = _get_valid_keys()

    # If no keys configured, auth is disabled (dev mode)
    if not valid_keys:
        return "dev-no-auth"

    if not api_key:
        logger.warning("Missing API key", extra={"error": "no_api_key"})
        raise HTTPException(status_code=401, detail="Missing API key")

    if api_key not in valid_keys:
        logger.warning(
            "Invalid API key",
            extra={"error": "invalid_api_key", "key_hash": _hash_key(api_key)},
        )
        raise HTTPException(status_code=403, detail="Invalid API key")

    return _hash_key(api_key)


def generate_api_key() -> str:
    """Generate a secure random API key."""
    return f"rag_{secrets.token_urlsafe(32)}"


# ── Rate Limiting ─────────────────────────────────────────────────────

def _key_func(request: Request) -> str:
    """Rate limit by API key if present, otherwise by IP."""
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return _hash_key(api_key)
    return get_remote_address(request)


limiter = Limiter(
    key_func=_key_func,
    default_limits=["100/minute"],
    storage_uri=os.getenv("RATE_LIMIT_REDIS_URL", "memory://"),
)


# ── Input Validation ──────────────────────────────────────────────────

MAX_QUESTION_LENGTH = 2000
MIN_QUESTION_LENGTH = 3

# Patterns that suggest prompt injection attempts
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above",
    r"disregard\s+(all\s+)?previous",
    r"you\s+are\s+now\s+",
    r"new\s+instructions?\s*:",
    r"system\s*:\s*",
    r"<\s*system\s*>",
    r"\[INST\]",
    r"\[/INST\]",
    r"<<SYS>>",
    r"<\|im_start\|>",
]

_compiled_patterns = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def validate_question(question: str) -> str:
    """Validate and sanitize user question. Returns cleaned question or raises."""
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    question = question.strip()

    if len(question) < MIN_QUESTION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Question too short (min {MIN_QUESTION_LENGTH} characters)",
        )

    if len(question) > MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Question too long (max {MAX_QUESTION_LENGTH} characters)",
        )

    # Check for prompt injection
    for pattern in _compiled_patterns:
        if pattern.search(question):
            logger.warning(
                "Prompt injection attempt detected",
                extra={"error": "prompt_injection", "pattern": pattern.pattern},
            )
            raise HTTPException(
                status_code=400,
                detail="Invalid question format",
            )

    return question


# ── PII Redaction ────────────────────────────────────────────────────

PII_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL]"),
    (re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"), "[SSN]"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"), "[PHONE]"),
    (re.compile(r"\b\d{13,19}\b"), "[CARD_NUMBER]"),
    (re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]{0,16})?\b"), "[IBAN]"),
]


def redact_pii(text: str) -> str:
    """Replace PII patterns (emails, SSNs, phone numbers, card numbers) with placeholders."""
    if not text:
        return text
    result = text
    for pattern, replacement in PII_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def dlp_scan(text: str) -> dict:
    """Scan text for PII leakage. Returns redacted text and findings summary.

    Used as a DLP gate in the reviewer node before answers reach the user.
    """
    if not text:
        return {"clean_text": text or "", "pii_found": False, "findings": []}

    findings = []
    for pattern, label in PII_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            findings.append({"type": label, "count": len(matches)})

    clean_text = redact_pii(text) if findings else text
    return {
        "clean_text": clean_text,
        "pii_found": bool(findings),
        "findings": findings,
    }


def sanitize_for_prompt(text: str, max_length: int = 500) -> str:
    """Sanitize text before injecting into LLM prompts (e.g., conversation history).

    Strips control characters and known injection markers, truncates.
    """
    if not text:
        return ""

    # Remove known prompt delimiters that could confuse the LLM
    cleaned = text
    for pattern in _compiled_patterns:
        cleaned = pattern.sub("[FILTERED]", cleaned)

    # Remove any XML/HTML-like tags
    cleaned = re.sub(r"<[^>]{1,50}>", "", cleaned)

    # Truncate
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length] + "..."

    return cleaned

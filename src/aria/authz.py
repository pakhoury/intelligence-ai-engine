"""
Authorization client — delegates identity resolution and RBAC/ABAC decisions
to an external Authorization service instead of maintaining a local
role/permission model.

This follows the Policy Enforcement Point / Policy Decision Point split used
by OPA, AWS Verified Permissions, Zanzibar-style authorizers, and NIST
SP 800-162 ABAC guidance: this module (the PEP) gathers the presented
credential and whatever resource attributes it already owns (e.g. who
created a session, per workflow.py's checkpointed state), and hands the
allow/deny decision to the external PDP. No roles or permissions are
hardcoded in this codebase — the RBAC service owns that model entirely.

Wire protocol — POST {AUTHZ_SERVICE_URL}/v1/authorize:

    Request:
        {
          "credential": "<raw bearer/API key from the incoming request>",
          "action": "query:submit" | "audit:read",
          "resource": {
            "type": "thread",
            "id": "<thread_id>",
            "owner_id": "<principal.sub that created this thread, or null>",
            "tenant_id": "<owning tenant, or null>"
          } | null
        }

    Response (200):
        {
          "allow": true,
          "principal": {
            "sub": "user-123",
            "tenant_id": "org-456",
            "roles": ["analyst", "auditor"]
          },
          "reason": "role:analyst has query:submit"
        }

Any non-200 response, malformed JSON, timeout, or open circuit is treated as
a deny. This is an authorization gate, not the LLM call path — unlike
llm_circuit_breaker (which fails over to a fallback model), failures here
must fail CLOSED, never open.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import httpx

from circuit_breaker import CircuitBreaker, CircuitBreakerOpenError
from observability import logger


def _default_timeout() -> float:
    return float(os.getenv("AUTHZ_TIMEOUT", "3"))


@dataclass(frozen=True)
class Principal:
    """Identity + roles as resolved by the external RBAC service. This app
    never assigns roles itself — it only reads what the PDP returned.
    """
    sub: str
    tenant_id: str | None = None
    roles: tuple[str, ...] = ()


@dataclass(frozen=True)
class AuthzDecision:
    allowed: bool
    principal: Principal | None
    reason: str = ""


class AuthzClient(Protocol):
    async def authorize(
        self, credential: str | None, action: str, resource: dict | None = None,
    ) -> AuthzDecision: ...


# Reuses the same thread-safe CircuitBreaker as the LLM path (circuit_breaker.py),
# which already reports rag_circuit_breaker_state{name="authz"} and
# rag_circuit_breaker_trips_total{name="authz"} to Prometheus for free.
authz_circuit_breaker = CircuitBreaker(
    name="authz", failure_threshold=5, recovery_timeout=15.0, success_threshold=2,
)


class HttpAuthzClient:
    """Calls an external PDP over HTTP(S). Fails closed on any error."""

    def __init__(self, base_url: str, timeout: float | None = None):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout if timeout is not None else _default_timeout()

    async def authorize(
        self, credential: str | None, action: str, resource: dict | None = None,
    ) -> AuthzDecision:
        try:
            authz_circuit_breaker.before_call()
        except CircuitBreakerOpenError:
            logger.warning(
                "Authz circuit open, denying by default",
                extra={"node": "authz", "action": action, "error": "authz_circuit_open"},
            )
            return AuthzDecision(allowed=False, principal=None, reason="authz_service_unavailable")

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/v1/authorize",
                    json={"credential": credential, "action": action, "resource": resource},
                )
            response.raise_for_status()
            body = response.json()
            authz_circuit_breaker.record_success()
        except Exception as e:
            authz_circuit_breaker.record_failure()
            logger.error(
                f"Authz service call failed, denying by default: {e}",
                extra={"node": "authz", "action": action, "error": str(e)},
            )
            return AuthzDecision(allowed=False, principal=None, reason="authz_service_error")

        if not isinstance(body, dict) or "allow" not in body:
            logger.error(
                "Authz service returned a malformed decision, denying by default",
                extra={"node": "authz", "action": action, "error": "malformed_authz_response"},
            )
            return AuthzDecision(allowed=False, principal=None, reason="malformed_authz_response")

        principal = None
        raw_principal = body.get("principal")
        if isinstance(raw_principal, dict) and raw_principal.get("sub"):
            principal = Principal(
                sub=str(raw_principal["sub"]),
                tenant_id=raw_principal.get("tenant_id"),
                roles=tuple(raw_principal.get("roles", [])),
            )

        return AuthzDecision(
            allowed=bool(body["allow"]),
            principal=principal,
            reason=body.get("reason", ""),
        )


class DevStubAuthzClient:
    """Allow-everything stub for local development / CI when no
    AUTHZ_SERVICE_URL is configured. Never selected outside
    ENVIRONMENT=development — see get_authz_client().
    """

    async def authorize(
        self, credential: str | None, action: str, resource: dict | None = None,
    ) -> AuthzDecision:
        return AuthzDecision(
            allowed=True,
            principal=Principal(sub="dev-user", tenant_id="dev-tenant", roles=("dev-admin",)),
            reason="dev_stub_allow_all",
        )


_client: AuthzClient | None = None


def get_authz_client() -> AuthzClient:
    """Same fail-closed-outside-dev pattern used elsewhere in this codebase
    (security.verify_api_key, workflow.create_persistent_app): an
    unconfigured authorization dependency must never silently allow
    everything once the app leaves development.
    """
    global _client
    if _client is not None:
        return _client

    service_url = os.getenv("AUTHZ_SERVICE_URL", "")
    if service_url:
        _client = HttpAuthzClient(service_url)
    elif os.getenv("ENVIRONMENT", "development") == "development":
        logger.warning(
            "AUTHZ_SERVICE_URL not set — using allow-all dev stub",
            extra={"node": "authz"},
        )
        _client = DevStubAuthzClient()
    else:
        raise RuntimeError(
            "AUTHZ_SERVICE_URL is not configured in a non-development environment. "
            "Authorization cannot fail open — set AUTHZ_SERVICE_URL or ENVIRONMENT=development."
        )
    return _client


def reset_client_cache() -> None:
    """Test hook — forces get_authz_client() to re-read env vars."""
    global _client
    _client = None

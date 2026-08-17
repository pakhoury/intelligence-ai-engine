"""
Unit tests for the authz module — the PEP that delegates identity/RBAC
decisions to an external service. All external calls are mocked; these tests
verify the fail-closed contract, not any real PDP.
"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "aria"))

import authz
from authz import (
    DevStubAuthzClient,
    HttpAuthzClient,
    get_authz_client,
    reset_client_cache,
)
from circuit_breaker import CircuitState


@pytest.fixture(autouse=True)
def _reset_authz_state():
    reset_client_cache()
    authz.authz_circuit_breaker.reset()
    yield
    reset_client_cache()
    authz.authz_circuit_breaker.reset()


def _mock_response(status_code=200, json_body=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_body or {}
    if status_code >= 400:
        import httpx
        resp.raise_for_status.side_effect = httpx.HTTPStatusError(
            "error", request=MagicMock(), response=resp,
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


class TestHttpAuthzClientAllow:
    async def test_allow_with_principal(self):
        client = HttpAuthzClient("http://authz.internal")
        body = {
            "allow": True,
            "principal": {"sub": "user-1", "tenant_id": "org-1", "roles": ["analyst"]},
            "reason": "role:analyst has query:submit",
        }
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _mock_response(200, body)
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            decision = await client.authorize("key1", "query:submit")

        assert decision.allowed is True
        assert decision.principal.sub == "user-1"
        assert decision.principal.tenant_id == "org-1"
        assert decision.principal.roles == ("analyst",)

    async def test_deny_with_reason(self):
        client = HttpAuthzClient("http://authz.internal")
        body = {"allow": False, "principal": None, "reason": "not_owner"}
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _mock_response(200, body)
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            decision = await client.authorize("key1", "audit:read", resource={"type": "thread", "id": "t1"})

        assert decision.allowed is False
        assert decision.principal is None
        assert decision.reason == "not_owner"


class TestHttpAuthzClientFailsClosed:
    async def test_http_error_denies(self):
        client = HttpAuthzClient("http://authz.internal")
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _mock_response(500)
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            decision = await client.authorize("key1", "query:submit")

        assert decision.allowed is False
        assert decision.principal is None

    async def test_connection_error_denies(self):
        client = HttpAuthzClient("http://authz.internal")
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = ConnectionError("unreachable")
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            decision = await client.authorize("key1", "query:submit")

        assert decision.allowed is False

    async def test_malformed_response_denies(self):
        client = HttpAuthzClient("http://authz.internal")
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.return_value = _mock_response(200, {"unexpected": "shape"})
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            decision = await client.authorize("key1", "query:submit")

        assert decision.allowed is False
        assert decision.reason == "malformed_authz_response"

    async def test_circuit_opens_and_denies_without_calling_service(self):
        """After enough failures the breaker trips; further calls must deny
        immediately without attempting another network call (fail closed,
        not fail open, unlike the LLM fallback path)."""
        client = HttpAuthzClient("http://authz.internal")
        with patch("httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.post.side_effect = ConnectionError("unreachable")
            mock_client_cls.return_value.__aenter__.return_value = mock_client

            for _ in range(5):
                decision = await client.authorize("key1", "query:submit")
                assert decision.allowed is False

            assert authz.authz_circuit_breaker.state == CircuitState.OPEN

            call_count_before = mock_client.post.call_count
            decision = await client.authorize("key1", "query:submit")
            assert decision.allowed is False
            assert decision.reason == "authz_service_unavailable"
            # Breaker was open — no additional network call was attempted.
            assert mock_client.post.call_count == call_count_before


class TestDevStubAuthzClient:
    async def test_allows_everything(self):
        stub = DevStubAuthzClient()
        decision = await stub.authorize("anything", "query:submit")
        assert decision.allowed is True
        assert decision.principal.sub == "dev-user"
        assert "dev-admin" in decision.principal.roles


class TestGetAuthzClient:
    def test_uses_http_client_when_configured(self):
        with patch.dict(os.environ, {"AUTHZ_SERVICE_URL": "http://authz.internal"}):
            client = get_authz_client()
        assert isinstance(client, HttpAuthzClient)

    def test_uses_dev_stub_when_unconfigured_in_dev(self):
        with patch.dict(os.environ, {"AUTHZ_SERVICE_URL": "", "ENVIRONMENT": "development"}):
            client = get_authz_client()
        assert isinstance(client, DevStubAuthzClient)

    def test_fails_closed_when_unconfigured_in_production(self):
        with (
            patch.dict(os.environ, {"AUTHZ_SERVICE_URL": "", "ENVIRONMENT": "production"}),
            pytest.raises(RuntimeError),
        ):
            get_authz_client()

    def test_client_is_cached(self):
        with patch.dict(os.environ, {"AUTHZ_SERVICE_URL": "", "ENVIRONMENT": "development"}):
            first = get_authz_client()
            second = get_authz_client()
        assert first is second

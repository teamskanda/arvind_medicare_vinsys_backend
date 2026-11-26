"""API contract tests: route protection and the error envelope.

These drive the real ASGI app via an in-process transport. Lifespan is not
run, so no database connection is opened — every assertion here concerns
behaviour that resolves before any query.
"""

from __future__ import annotations

import uuid

import httpx
import jwt
import pytest

import main

CHATBOT_ID = str(uuid.uuid4())
PAGE_ID = str(uuid.uuid4())
CONVERSATION_ID = str(uuid.uuid4())
ASSET_ID = str(uuid.uuid4())

# Every endpoint that must reject an anonymous caller.
PROTECTED_ENDPOINTS = [
    ("GET", "/api/auth/me"),
    ("GET", "/api/chatbots"),
    ("POST", "/api/chatbots"),
    ("GET", f"/api/chatbots/{CHATBOT_ID}"),
    ("PATCH", f"/api/chatbots/{CHATBOT_ID}"),
    ("DELETE", f"/api/chatbots/{CHATBOT_ID}"),
    ("POST", f"/api/chatbots/{CHATBOT_ID}/crawl"),
    ("POST", f"/api/chatbots/{CHATBOT_ID}/crawl/retry"),
    ("GET", f"/api/chatbots/{CHATBOT_ID}/crawl/status"),
    ("GET", f"/api/chatbots/{CHATBOT_ID}/pages"),
    ("GET", f"/api/chatbots/{CHATBOT_ID}/pages/{PAGE_ID}"),
    ("PATCH", f"/api/chatbots/{CHATBOT_ID}/pages/{PAGE_ID}"),
    # Phase 2 — the AI engine. These reach the model and the knowledge base,
    # so an auth gap here is worse than on the Phase 1 endpoints.
    ("GET", f"/api/chatbots/{CHATBOT_ID}/chat/config"),
    ("POST", f"/api/chatbots/{CHATBOT_ID}/chat"),
    ("POST", f"/api/chatbots/{CHATBOT_ID}/conversations"),
    ("GET", f"/api/chatbots/{CHATBOT_ID}/conversations"),
    (
        "GET",
        f"/api/chatbots/{CHATBOT_ID}/conversations/{CONVERSATION_ID}/messages",
    ),
    # Phase 3 — deployment management. Publishing makes a chatbot reachable by
    # the whole internet, so an anonymous caller must never be able to do it.
    ("GET", f"/api/chatbots/{CHATBOT_ID}/deployment"),
    ("POST", f"/api/chatbots/{CHATBOT_ID}/publish"),
    ("POST", f"/api/chatbots/{CHATBOT_ID}/unpublish"),
    # Phase 4 — the customisation studio. These read and write one
    # customer's design, and the upload endpoint accepts bytes we then
    # serve, so an anonymous caller must reach none of them.
    ("GET", f"/api/chatbots/{CHATBOT_ID}/customization"),
    ("PUT", f"/api/chatbots/{CHATBOT_ID}/customization"),
    ("POST", f"/api/chatbots/{CHATBOT_ID}/customization/preview"),
    ("POST", f"/api/chatbots/{CHATBOT_ID}/customization/chat"),
    ("POST", f"/api/chatbots/{CHATBOT_ID}/customization/reset"),
    ("POST", f"/api/chatbots/{CHATBOT_ID}/assets"),
    ("DELETE", f"/api/chatbots/{CHATBOT_ID}/assets/{ASSET_ID}"),
]

# Endpoints that are anonymous *on purpose*. The guarantee for these is the
# opposite one: they must NOT demand a Supabase token, because a website
# visitor has no account. Their own gates are publication status, the signed
# visitor session, the allowed-domain check and the request limits.
PUBLIC_ENDPOINTS = [
    ("GET", "/api/public/bots/{public_id}/config"),
    ("POST", "/api/public/bots/{public_id}/sessions"),
    ("POST", "/api/public/bots/{public_id}/chat"),
    ("GET", "/w/{public_id}.js"),
    # Phase 4: the avatar in a chat window on a stranger's screen.
    ("GET", "/api/public/assets/{asset_id}"),
]


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(
        transport=transport, base_url="http://testserver"
    ) as http_client:
        yield http_client


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


async def test_health_is_public(client: httpx.AsyncClient):
    response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Authentication gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("method", "path"), PROTECTED_ENDPOINTS)
async def test_protected_endpoints_reject_anonymous_callers(
    client: httpx.AsyncClient, method: str, path: str
):
    response = await client.request(method, path, json={})
    assert response.status_code == 401, f"{method} {path} should require auth"

    body = response.json()
    assert "error" in body, "errors must use the standard envelope"
    assert body["error"]["code"] == "unauthenticated"
    # The message is written for a customer, not a developer.
    assert "sign in" in body["error"]["message"].lower()


@pytest.mark.parametrize(("method", "path"), PROTECTED_ENDPOINTS)
async def test_protected_endpoints_reject_malformed_tokens(
    client: httpx.AsyncClient, method: str, path: str
):
    response = await client.request(
        method, path, json={}, headers={"Authorization": "Bearer not-a-real-jwt"}
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthenticated"


async def test_unsupported_signing_algorithm_is_rejected(client: httpx.AsyncClient):
    """An unsigned ('alg': 'none') token must never be accepted."""
    token = jwt.encode({"sub": str(uuid.uuid4())}, key="", algorithm="none")
    response = await client.get(
        "/api/chatbots", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthenticated"


async def test_hs256_token_without_configured_secret_is_not_accepted(
    client: httpx.AsyncClient,
):
    """A legacy-style token cannot authenticate when no secret is configured.

    The caller must not be let in, and the response must not disclose the
    server-side cause beyond a generic message.
    """
    token = jwt.encode(
        {
            "sub": str(uuid.uuid4()),
            "email": "attacker@example.com",
            "aud": "authenticated",
            "iss": "https://testproject.supabase.co/auth/v1",
            "exp": 9_999_999_999,
        },
        key="guessed-signing-secret-with-at-least-32-bytes",
        algorithm="HS256",
    )
    response = await client.get(
        "/api/chatbots", headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code in (401, 500)
    body = response.json()
    assert "error" in body
    # No stack trace, SQL or key material in the message.
    assert "guessed-secret" not in body["error"]["message"]
    assert "Traceback" not in body["error"]["message"]


async def test_empty_bearer_token_is_rejected(client: httpx.AsyncClient):
    response = await client.get("/api/chatbots", headers={"Authorization": "Bearer "})
    assert response.status_code == 401


async def test_non_bearer_scheme_is_rejected(client: httpx.AsyncClient):
    response = await client.get(
        "/api/chatbots", headers={"Authorization": "Basic dXNlcjpwYXNz"}
    )
    assert response.status_code == 401


# ---------------------------------------------------------------------------
# Error envelope consistency
# ---------------------------------------------------------------------------


async def test_unknown_route_uses_the_error_envelope(client: httpx.AsyncClient):
    response = await client.get("/api/does-not-exist")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "not_found"


async def test_method_not_allowed_uses_the_error_envelope(client: httpx.AsyncClient):
    response = await client.put("/api/health")
    assert response.status_code == 405
    assert "error" in response.json()


async def test_openapi_schema_is_generated(client: httpx.AsyncClient):
    response = await client.get("/openapi.json")
    assert response.status_code == 200
    paths = response.json()["paths"]
    for _, path in PROTECTED_ENDPOINTS:
        template = (
            path.replace(CHATBOT_ID, "{chatbot_id}")
            .replace(PAGE_ID, "{page_id}")
            .replace(CONVERSATION_ID, "{conversation_id}")
            .replace(ASSET_ID, "{asset_id}")
        )
        assert template in paths, f"{template} missing from the OpenAPI schema"


# ---------------------------------------------------------------------------
# Phase 2 — the streaming endpoint must also be gated
# ---------------------------------------------------------------------------


async def test_the_streaming_chat_endpoint_rejects_anonymous_callers(
    client: httpx.AsyncClient,
):
    """Streaming responds 200 once open, so the gate must close before that."""
    response = await client.post(
        f"/api/chatbots/{CHATBOT_ID}/chat?stream=true",
        json={"message": "hello"},
    )
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "unauthenticated"
    assert "text/event-stream" not in response.headers.get("content-type", "")


async def test_an_oversized_question_is_rejected_before_auth_work(
    client: httpx.AsyncClient,
):
    """The length cap lives on the schema, so it is enforced at the edge."""
    response = await client.post(
        f"/api/chatbots/{CHATBOT_ID}/chat",
        json={"message": "x" * 5_000},
    )
    # Unauthenticated, so 401 wins; the point is that it is never a 500.
    assert response.status_code in (401, 422)
    assert "error" in response.json()


# ---------------------------------------------------------------------------
# No secrets in the public schema
# ---------------------------------------------------------------------------


async def test_the_openapi_schema_contains_no_credentials(client: httpx.AsyncClient):
    """The schema is served publicly; it must describe shapes, not secrets."""
    response = await client.get("/openapi.json")
    body = response.text.lower()
    for secret in (
        "fc-test-key",
        "gemma_api_key",
        "supabase_jwt_secret",
        "database_url",
        "visitor_session_secret",
    ):
        assert secret not in body, f"/openapi.json mentions {secret!r}"


# ---------------------------------------------------------------------------
# Phase 3 — the public surface is anonymous by design
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("method", "template"), PUBLIC_ENDPOINTS)
async def test_public_endpoints_are_registered(
    client: httpx.AsyncClient, method: str, template: str
):
    response = await client.get("/openapi.json")
    assert template in response.json()["paths"], f"{template} is not routed"


@pytest.mark.parametrize(("method", "template"), PUBLIC_ENDPOINTS)
async def test_public_endpoints_do_not_demand_a_supabase_token(
    client: httpx.AsyncClient, method: str, template: str
):
    """A website visitor has no account, so a 401 here would break the widget.

    A malformed handle is used so the request is refused on its shape before
    any database access — which also proves the refusal is not an auth gate.
    """
    path = template.replace("{public_id}", "not-a-valid-handle").replace(
        "{asset_id}", "not-a-uuid"
    )
    response = await client.request(method, path, json={"message": "hello"})
    assert response.status_code != 401, f"{method} {path} wrongly requires auth"


@pytest.mark.parametrize(
    "handle",
    [
        "not-a-valid-handle",  # hyphens are not in the handle alphabet
        "x" * 100,  # too long
        "has_underscore",
        "has.dot",
    ],
)
async def test_a_malformed_public_handle_is_refused_without_a_database_call(
    client: httpx.AsyncClient, handle: str
):
    """The shape check runs first, so garbage never reaches a query."""
    response = await client.get(f"/api/public/bots/{handle}/config")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "chatbot_unavailable"
    # Identical wording for unknown, draft and unpublished: the endpoint must
    # not let anyone enumerate which handles exist.
    assert "not available" in body["error"]["message"]


async def test_the_widget_loader_serves_a_no_op_for_an_unknown_handle(
    client: httpx.AsyncClient,
):
    """A 404 would log an error on the customer's own website."""
    response = await client.get("/w/not-a-valid-handle.js")
    assert response.status_code == 200
    assert "javascript" in response.headers["content-type"]
    assert response.headers["cache-control"] == "no-store"
    assert "not published" in response.text
    # A no-op: no iframe, no launcher, nothing drawn.
    assert "createElement" not in response.text


async def test_an_oversized_public_question_is_rejected_at_the_edge(
    client: httpx.AsyncClient,
):
    """The public cap is lower than the playground's, and schema-enforced."""
    response = await client.post(
        "/api/public/bots/abc234defghijkmnpqrstu/chat",
        json={"message": "x" * 5_000},
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "validation_error"


async def test_the_public_chat_schema_does_not_accept_a_chatbot_id(
    client: httpx.AsyncClient,
):
    """A visitor must not be able to address a chatbot by internal id.

    The only identifier a public endpoint takes is the path's `public_id`, so
    there is no body field to tamper with. An extra key is ignored by the
    schema rather than honoured.
    """
    schema = (await client.get("/openapi.json")).json()
    body = schema["components"]["schemas"]["PublicChatRequest"]["properties"]
    assert set(body) == {"message", "conversation_id"}
    assert "chatbot_id" not in body
    assert "owner_id" not in body
    assert "session_id" not in body

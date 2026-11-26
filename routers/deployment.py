"""Deployment endpoints: publication, the public chatbot API, and the loader.

Three routers in one module because they are three faces of one feature, and
keeping them together makes the trust boundary between them visible:

  * `router`        — authenticated. Mounted at /api/chatbots. Every endpoint
                      requires a verified Supabase token and goes through
                      helpers that filter on `owner_id`.
  * `public_router` — anonymous. Mounted at /api/public. No token, no owner,
                      no customer data beyond the allowlisted visual config.
                      Each endpoint resolves the chatbot by `public_id` with
                      a publication predicate before anything else happens.
  * `widget_router` — anonymous. Mounted at /w. Serves generated JavaScript.

The rule that keeps this safe: a public endpoint never accepts an internal
identifier. It takes `public_id` and nothing else, so there is no parameter a
visitor could tamper with to address a different customer's chatbot, and no
code path where a chatbot id arrives from the request body.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from helpers import chat_engine, customization, deployment, errors, visitor_session, widget
from helpers import chatbots as chatbot_helpers
from helpers import conversations as conversation_helpers
from helpers.auth import CurrentUser
from helpers.database import get_db
from models.chatbot import Chatbot
from models.deployment import (
    DeploymentInfoOut,
    PublicBotConfigOut,
    PublicChatRequest,
    PublishRequest,
    VisitorSessionOut,
)
from models.message import ChatResponse
from models.user import AuthenticatedUser

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chatbots", tags=["deployment"])
public_router = APIRouter(prefix="/api/public", tags=["public"])
widget_router = APIRouter(tags=["public"])

DbSession = Annotated[AsyncSession, Depends(get_db)]

# The visitor session travels in a header, not a cookie. The chatbot runs in a
# third-party iframe, and every current browser blocks cookies in that
# context by default — so a cookie-based session would simply not work on the
# customers' sites this feature exists to serve.
VisitorToken = Annotated[str | None, Header(alias="X-Visitor-Token")]


def _client_ip(request: Request) -> str | None:
    """Best-effort client address for the per-IP request limit.

    `X-Forwarded-For` is only consulted when the deployment is behind a proxy
    we control; its left-most entry is the original client. It is trivially
    forgeable by a direct caller, which is why the per-IP limit is documented
    as the weakest of the three and the per-chatbot limit is the backstop.
    The value is hashed before use and never stored.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        first = forwarded.split(",")[0].strip()
        if first:
            return first[:64]
    return request.client.host if request.client else None


# ===========================================================================
# Authenticated: publication and deployment information
# ===========================================================================


async def _deployment_info(
    db: AsyncSession, user: AuthenticatedUser, chatbot: Chatbot
) -> DeploymentInfoOut:
    settings = get_settings()
    launcher = deployment.resolve_launcher(chatbot, published=False)
    readiness = await deployment.publish_readiness(db, user, chatbot)
    ai = deployment.ai_config_for(chatbot, published=False)

    return DeploymentInfoOut(
        chatbot_id=chatbot.id,
        public_id=chatbot.public_id,
        chatbot_name=ai.chatbot_name,
        website_url=chatbot.website_url,
        status=chatbot.status,
        is_published=chatbot.status == "published",
        published_at=chatbot.published_at,
        unpublished_at=chatbot.unpublished_at,
        can_publish=readiness.can_publish,
        publish_blocked_reason=readiness.reason,
        included_pages=readiness.included_pages,
        ai_ready=settings.gemma_configured,
        has_unpublished_changes=customization.has_unpublished_changes(chatbot),
        embed_url=settings.embed_url(chatbot.public_id),
        panel_url=settings.embed_url(chatbot.public_id, mode="panel"),
        standalone_url=settings.embed_url(chatbot.public_id, mode="full"),
        script_url=settings.widget_script_url(chatbot.public_id),
        allowed_domains=list(chatbot.allowed_domains or []),
        launcher=launcher,
        snippets=widget.build_snippets(chatbot, launcher),
        platforms=widget.platform_instructions(),
    )


@router.get("/{chatbot_id}/deployment", response_model=DeploymentInfoOut)
async def get_deployment(
    chatbot_id: uuid.UUID, user: CurrentUser, db: DbSession
) -> DeploymentInfoOut:
    """Everything the Deploy tab renders: URLs, snippets and publish state.

    Available before publishing on purpose — the customer should be able to
    read the install instructions and see their URL while deciding.
    """
    chatbot = await chatbot_helpers.get_chatbot(db, user, chatbot_id)
    return await _deployment_info(db, user, chatbot)


@router.post("/{chatbot_id}/publish", response_model=DeploymentInfoOut)
async def publish_chatbot(
    chatbot_id: uuid.UUID,
    user: CurrentUser,
    db: DbSession,
    payload: PublishRequest | None = None,
) -> DeploymentInfoOut:
    """Make the chatbot live at its permanent embed URL."""
    chatbot = await deployment.publish(db, user, chatbot_id, payload)
    return await _deployment_info(db, user, chatbot)


@router.post("/{chatbot_id}/unpublish", response_model=DeploymentInfoOut)
async def unpublish_chatbot(
    chatbot_id: uuid.UUID, user: CurrentUser, db: DbSession
) -> DeploymentInfoOut:
    """Stop public conversations, keeping the URL, settings and knowledge base."""
    chatbot = await deployment.unpublish(db, user, chatbot_id)
    return await _deployment_info(db, user, chatbot)


# ===========================================================================
# Public: the chatbot anonymous visitors talk to
# ===========================================================================


@public_router.get("/bots/{public_id}/config", response_model=PublicBotConfigOut)
async def get_public_config(
    public_id: str, request: Request, db: DbSession
) -> PublicBotConfigOut:
    """The allowlisted visual configuration for one published chatbot.

    This is the first call the embed page makes. It is also the enforcement
    point for "unpublished chatbots must not accept public conversations":
    an unpublished handle 404s here, so the page shows the unavailable state
    without ever reaching the chat endpoint.
    """
    chatbot = await deployment.get_published_chatbot(db, public_id)
    deployment.check_embedding_origin(
        chatbot,
        embed_origin=request.headers.get("x-embed-origin"),
        origin=request.headers.get("origin"),
        referer=request.headers.get("referer"),
    )
    return deployment.public_config(chatbot)


@public_router.get('/bots/{public_id}/frame-policy')
async def get_frame_policy(public_id: str, db: DbSession) -> JSONResponse:
    """Only public embedding policy, for the Next.js response-header middleware."""
    chatbot = await deployment.get_published_chatbot(db, public_id)
    return JSONResponse({'frame_ancestors': deployment.frame_ancestors(chatbot)}, headers={'Cache-Control': 'no-store'})


@public_router.get("/bots/{public_id}/widget-config")
async def get_widget_config(public_id: str, request: Request, db: DbSession) -> JSONResponse:
    try:
        chatbot = await deployment.get_published_chatbot(db, public_id)
    except errors.AppError as exc:
        if exc.status_code != 404:
            raise
        return JSONResponse(
            {"error": {"code": "not_found", "message": "Assistant unavailable."}},
            status_code=404,
            headers={"Cache-Control": "no-store", "Access-Control-Allow-Origin": "*"},
        )
    deployment.check_embedding_origin(
        chatbot, origin=request.headers.get("origin"), referer=request.headers.get("referer"),
    )
    appearance = customization.resolve_appearance(chatbot, published=True)
    return JSONResponse(
        widget.build_loader_config(
            chatbot, deployment.resolve_launcher(chatbot, published=True),
            window_radius=appearance.border_radius,
        ),
        headers={"Cache-Control": "no-store", "Access-Control-Allow-Origin": "*"},
    )


@public_router.post("/bots/{public_id}/sessions", response_model=VisitorSessionOut)
async def create_visitor_session(
    public_id: str, request: Request, db: DbSession
) -> VisitorSessionOut:
    """Issue an anonymous visitor session token.

    The token is bound to this chatbot and expires, and the session handle
    inside it is what isolates one visitor's conversation from another's.
    Nothing is written to the database: see helpers/visitor_session.py for
    why the session is stateless.
    """
    chatbot = await deployment.get_published_chatbot(db, public_id)
    deployment.check_embedding_origin(
        chatbot,
        embed_origin=request.headers.get("x-embed-origin"),
        origin=request.headers.get("origin"),
        referer=request.headers.get("referer"),
    )

    # A session costs nothing to mint, but minting them in a loop is a cheap
    # way to probe. Counted per IP alongside the chat limit.
    conversation_helpers.enforce_ip_rate_limit(_client_ip(request))

    token, session = visitor_session.issue(chatbot.public_id)
    return VisitorSessionOut(
        token=token,
        expires_at=datetime.fromtimestamp(session.expires_at, tz=timezone.utc),
    )


@public_router.post("/bots/{public_id}/chat", response_model=ChatResponse)
async def public_chat(
    public_id: str,
    payload: PublicChatRequest,
    request: Request,
    db: DbSession,
    visitor_token: VisitorToken = None,
    stream: Annotated[bool, Query(description="Stream the reply as SSE.")] = False,
) -> ChatResponse | StreamingResponse:
    """Answer a visitor's question using the existing Gemma engine.

    Four gates before a single token is spent, in this order:

      1. the handle must resolve to a chatbot whose status is `published`;
      2. the embedding origin must be authorised, when the customer set a list;
      3. the visitor session token must verify and be bound to this chatbot;
      4. the per-session, per-chatbot and per-IP limits must not be exhausted.

    Everything after that is `helpers/chat_engine.py` — the same retrieval,
    token budget, prompt construction, provider call and persistence the
    dashboard playground uses, reading the chatbot's *published*
    configuration.
    """
    settings = get_settings()

    chatbot = await deployment.get_published_chatbot(db, public_id)
    deployment.check_embedding_origin(
        chatbot,
        embed_origin=request.headers.get("x-embed-origin"),
        origin=request.headers.get("origin"),
        referer=request.headers.get("referer"),
    )
    session = visitor_session.verify(visitor_token, public_id=chatbot.public_id)

    # The configured ceiling, enforced in addition to the schema's hard cap so
    # an operator can tighten it without a code change.
    if len(payload.message) > settings.public_chat_max_message_length:
        raise errors.AppError(
            "message_too_long",
            "That message is too long. Please shorten it and try again.",
            status_code=422,
        )

    turn_request = await chat_engine.resolve_public_turn(
        db,
        chatbot,
        session,
        question=payload.message,
        conversation_id=payload.conversation_id,
        client_ip=_client_ip(request),
    )
    turn = await chat_engine.prepare(db, turn_request)

    if not stream:
        return await chat_engine.complete(db, turn)

    return StreamingResponse(
        chat_engine.sse_events(db, turn),
        media_type="text/event-stream",
        headers=chat_engine.SSE_HEADERS,
    )


# ===========================================================================
# Public: the floating widget loader
# ===========================================================================


@widget_router.get("/w/{public_id}.js", response_class=PlainTextResponse)
async def widget_loader(public_id: str, db: DbSession) -> PlainTextResponse:
    """The JavaScript a customer pastes into their website.

    Generated per chatbot so the launcher's appearance, position and window
    size come from stored configuration rather than being hardcoded. It holds
    no AI logic, no knowledge base and no private configuration — only the
    public handle and the launcher's looks. See helpers/widget.py.

    An unknown, draft or unpublished handle gets a no-op script rather than a
    404: identical responses mean the endpoint does not confirm which handles
    exist, and the customer's own site does not log a failed request merely
    because they unpublished their chatbot.
    """
    try:
        chatbot = await deployment.get_published_chatbot(db, public_id)
    except errors.AppError:
        body = widget.build_unavailable_script(public_id)
        return PlainTextResponse(
            body,
            media_type="application/javascript; charset=utf-8",
            headers={"Cache-Control": "no-store"},
        )

    launcher = deployment.resolve_launcher(chatbot, published=True)
    # The floating window's corners follow the chatbot's own corner radius, so
    # the frame and the chat inside it match.
    appearance = customization.resolve_appearance(chatbot, published=True)
    return PlainTextResponse(
        widget.build_loader_script(
            chatbot, launcher, window_radius=appearance.border_radius
        ),
        media_type="application/javascript; charset=utf-8",
        headers={
            # Fetch current published settings on every installation load;
            # the runtime also refreshes existing installations periodically.
            "Cache-Control": "no-store",
            # The loader is fetched by a <script> tag from arbitrary customer
            # sites. Script tags need no CORS, but some bundlers and CSP
            # setups fetch it with XHR instead, so allow that too.
            "Access-Control-Allow-Origin": "*",
        },
    )

"""Deployment helpers: launcher resolution, domain matching and the public allowlist.

No database. Chatbot rows are constructed in memory, which is enough for
everything here: these functions are pure given a chatbot row, and they are
where the public-facing decisions are made.

The allowlist tests are the ones that matter most. `public_config` is served
to anonymous visitors on the open internet, so a field appearing there that
should not is a direct disclosure of a customer's configuration.
"""

from __future__ import annotations

import uuid

import pytest

from config import Settings, get_settings
from helpers import deployment, errors
from models.base import ChatbotStatus
from models.chatbot import Chatbot
from models.customization import MAX_SUGGESTED_QUESTIONS

PUBLIC_ID = "abc234defghijkmnpqrstu"


def make_chatbot(
    *,
    draft: dict | None = None,
    published: dict | None = None,
    deployment_config: dict | None = None,
    allowed_domains: list | None = None,
    status: str = ChatbotStatus.READY.value,
    brand_color: str | None = None,
    favicon_url: str | None = None,
) -> Chatbot:
    return Chatbot(
        id=uuid.uuid4(),
        public_id=PUBLIC_ID,
        owner_id=uuid.uuid4(),
        name="Acme Interiors",
        website_url="https://acme.test/",
        status=status,
        crawl_status="completed",
        draft_config=draft if draft is not None else {},
        published_config=published,
        deployment_config=deployment_config if deployment_config is not None else {},
        allowed_domains=allowed_domains if allowed_domains is not None else [],
        brand_color=brand_color,
        favicon_url=favicon_url,
    )


# ---------------------------------------------------------------------------
# Launcher settings
# ---------------------------------------------------------------------------


def test_an_empty_configuration_yields_the_documented_defaults():
    """A chatbot nobody has customised must still render correctly."""
    launcher = deployment.resolve_launcher(make_chatbot())
    settings = get_settings()
    assert launcher.launcher_size == settings.widget_launcher_size == 56
    assert launcher.position == "bottom-right"
    assert launcher.offset_x == 20
    assert launcher.window_width == 400
    assert launcher.window_height == 600


def test_stored_launcher_settings_are_honoured():
    launcher = deployment.resolve_launcher(
        make_chatbot(
            deployment_config={
                "launcher_size": 64,
                "position": "bottom-left",
                "offset_x": 30,
                "window_width": 480,
                "icon": "robot",
                "open_on_load": True,
            }
        )
    )
    assert launcher.launcher_size == 64
    assert launcher.position == "bottom-left"
    assert launcher.offset_x == 30
    assert launcher.window_width == 480
    assert launcher.icon == "robot"
    assert launcher.open_on_load is True


@pytest.mark.parametrize(
    "stored",
    [
        {"launcher_size": "big"},
        {"launcher_size": None},
        {"launcher_size": True},  # bool is an int subclass; must not slip through
        {"position": "top-middle"},
        {"open_on_load": "yes"},
    ],
)
def test_malformed_launcher_values_fall_back_to_defaults(stored: dict):
    """This JSONB is written by the Phase 4 UI; it must not be able to break."""
    launcher = deployment.resolve_launcher(make_chatbot(deployment_config=stored))
    assert launcher.launcher_size == 56
    assert launcher.position == "bottom-right"
    assert launcher.open_on_load is False


def test_out_of_range_dimensions_do_not_break_the_deployment_view():
    """A 5000px launcher would cover the customer's page."""
    launcher = deployment.resolve_launcher(
        make_chatbot(deployment_config={"launcher_size": 5000, "window_width": 9999})
    )
    # Clamped to the maximum the studio offers, not reset to the default:
    # someone who asked for "as large as possible" gets the largest allowed
    # launcher, and the customer's page is still safe.
    assert launcher.launcher_size == 96
    assert launcher.window_width == 640


def test_the_brand_colour_becomes_the_accent_when_no_override_exists():
    launcher = deployment.resolve_launcher(make_chatbot(brand_color="#1f5a8a"))
    assert launcher.accent_color == "#1f5a8a"


@pytest.mark.parametrize(
    "colour",
    [
        "red",
        "rgb(0,0,0)",
        "#ggg",
        "url(javascript:alert(1))",
        "#7d2a47; background:url(x)",
        "",
    ],
)
def test_only_a_hex_colour_is_accepted_as_an_accent(colour: str):
    """The accent is interpolated into generated CSS and inline styles."""
    launcher = deployment.resolve_launcher(
        make_chatbot(deployment_config={"accent_color": colour})
    )
    assert launcher.accent_color == "#7d2a47"


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
        "http://insecure.test/logo.png",
        "not a url",
        123,
        None,
    ],
)
def test_an_unsafe_launcher_icon_url_is_dropped(url):
    """This value ends up in an <img src> inside the customer's own page."""
    launcher = deployment.resolve_launcher(
        make_chatbot(deployment_config={"icon_url": url})
    )
    assert launcher.icon_url is None


def test_an_https_launcher_icon_url_is_kept():
    launcher = deployment.resolve_launcher(
        make_chatbot(deployment_config={"icon_url": "https://cdn.acme.test/logo.png"})
    )
    assert launcher.icon_url == "https://cdn.acme.test/logo.png"


# ---------------------------------------------------------------------------
# Draft vs published configuration
# ---------------------------------------------------------------------------


def test_the_playground_reads_the_draft_configuration():
    chatbot = make_chatbot(
        draft={"ai": {"chatbot_name": "Draft Name"}},
        published={"ai": {"chatbot_name": "Published Name"}},
    )
    assert deployment.ai_config_for(chatbot, published=False).chatbot_name == "Draft Name"


def test_the_public_chatbot_reads_the_published_configuration():
    """The whole point of the draft/published split.

    A customer editing their persona must not change what visitors see until
    they publish again.
    """
    chatbot = make_chatbot(
        draft={"ai": {"chatbot_name": "Draft Name", "persona": "Experimental"}},
        published={"ai": {"chatbot_name": "Published Name", "persona": "Formal"}},
    )
    config = deployment.ai_config_for(chatbot, published=True)
    assert config.chatbot_name == "Published Name"
    assert config.persona == "Formal"


def test_an_unpublished_chatbot_falls_back_to_defaults_not_to_the_draft():
    """`published_config` is null before the first publish.

    It must not quietly read the draft instead: that would mean a chatbot
    could serve unpublished settings.
    """
    chatbot = make_chatbot(draft={"ai": {"persona": "Secret experimental persona"}})
    config = deployment.ai_config_for(chatbot, published=True)
    assert config.persona == "Friendly and professional"
    assert "Secret" not in config.persona


def test_launcher_settings_also_split_draft_from_published():
    chatbot = make_chatbot(
        deployment_config={"launcher_size": 90},
        published={"deployment": {"launcher_size": 48}},
    )
    assert deployment.resolve_launcher(chatbot, published=False).launcher_size == 90
    assert deployment.resolve_launcher(chatbot, published=True).launcher_size == 48


# ---------------------------------------------------------------------------
# Suggested questions
# ---------------------------------------------------------------------------


def test_no_suggested_questions_by_default():
    """Inventing starter questions implies the chatbot can answer them."""
    assert deployment.suggested_questions(make_chatbot(), published=True) == []


def test_configured_suggested_questions_are_capped_and_cleaned():
    chatbot = make_chatbot(
        published={
            "ai": {
                "suggested_questions": [
                    "  What are your prices?  ",
                    "Where are you located?",
                    "Do you ship?",
                    "What is your warranty?",
                    "Are you open on Sundays?",
                    "Do you offer installation?",
                    "A seventh question, past the cap",
                    "Do you ship?",
                    42,
                    "",
                ]
            }
        }
    )
    questions = deployment.suggested_questions(chatbot, published=True)
    # Trimmed, de-duplicated, non-strings dropped, and capped at the number
    # the studio lets a customer enter.
    assert questions == [
        "What are your prices?",
        "Where are you located?",
        "Do you ship?",
        "What is your warranty?",
        "Are you open on Sundays?",
        "Do you offer installation?",
    ]
    assert len(questions) == MAX_SUGGESTED_QUESTIONS


def test_a_malformed_suggested_questions_value_is_ignored():
    chatbot = make_chatbot(published={"ai": {"suggested_questions": "not a list"}})
    assert deployment.suggested_questions(chatbot, published=True) == []


# ---------------------------------------------------------------------------
# The public allowlist — what a visitor is allowed to see
# ---------------------------------------------------------------------------


PRIVATE_CONFIG = {
    "ai": {
        "chatbot_name": "Acme Assistant",
        "greeting": "Hi there!",
        "persona": "Terse and slightly rude, internal note",
        "instructions": "Always upsell the premium package. Never mention refunds.",
        "fallback_message": "Ring the office on the private line.",
    },
    "deployment": {"accent_color": "#1f5a8a"},
}


def test_the_public_config_exposes_the_visible_essentials():
    chatbot = make_chatbot(published=PRIVATE_CONFIG, status=ChatbotStatus.PUBLISHED.value)
    config = deployment.public_config(chatbot)
    assert config.public_id == PUBLIC_ID
    assert config.chatbot_name == "Acme Assistant"
    assert config.greeting == "Hi there!"
    assert config.website_host == "acme.test"
    # The launcher keeps its own colour; the chat window's colours come from
    # the resolved appearance palettes.
    assert config.launcher.accent_color == "#1f5a8a"
    assert config.appearance.light.header_background == config.appearance.light.primary


def test_the_public_config_hides_the_persona_and_instructions():
    """The customer's business instructions are not the visitor's business."""
    chatbot = make_chatbot(published=PRIVATE_CONFIG, status=ChatbotStatus.PUBLISHED.value)
    serialised = deployment.public_config(chatbot).model_dump_json()
    assert "slightly rude" not in serialised
    assert "upsell" not in serialised
    assert "Never mention refunds" not in serialised


def test_the_public_config_hides_the_fallback_wording():
    """The fallback can contain a private phone number or an internal process."""
    chatbot = make_chatbot(published=PRIVATE_CONFIG, status=ChatbotStatus.PUBLISHED.value)
    serialised = deployment.public_config(chatbot).model_dump_json()
    assert "private line" not in serialised


def test_the_public_config_hides_internal_identifiers():
    """No owner id, no chatbot uuid: a visitor gets the public handle only."""
    chatbot = make_chatbot(published=PRIVATE_CONFIG, status=ChatbotStatus.PUBLISHED.value)
    serialised = deployment.public_config(chatbot).model_dump_json()
    assert str(chatbot.owner_id) not in serialised
    assert str(chatbot.id) not in serialised


def test_the_public_config_hides_the_allowed_domain_list():
    """Publishing the allowlist tells an attacker exactly what to forge."""
    chatbot = make_chatbot(
        published=PRIVATE_CONFIG,
        allowed_domains=["acme.test", "shop.acme.test"],
        status=ChatbotStatus.PUBLISHED.value,
    )
    serialised = deployment.public_config(chatbot).model_dump_json()
    assert "shop.acme.test" not in serialised


def test_the_public_config_field_set_is_an_explicit_allowlist():
    """A new chatbot field must not leak by being forgotten.

    This test fails when `PublicBotConfigOut` gains a field, which is the
    point: adding one should be a deliberate decision with a reviewer, not a
    side effect of adding a column.
    """
    chatbot = make_chatbot(published=PRIVATE_CONFIG, status=ChatbotStatus.PUBLISHED.value)
    assert set(deployment.public_config(chatbot).model_dump()) == {
        "public_id",
        "chatbot_name",
        "business_name",
        "greeting",
        "welcome_message",
        "website_host",
        "avatar_url",
        "logo_url",
        "cite_sources",
        "suggested_questions",
        "appearance",
        "launcher",
        "show_branding",
        "max_message_length",
    }


def test_an_insecure_favicon_is_not_offered_as_an_avatar():
    chatbot = make_chatbot(
        published=PRIVATE_CONFIG,
        favicon_url="http://acme.test/favicon.ico",
        status=ChatbotStatus.PUBLISHED.value,
    )
    assert deployment.public_config(chatbot).avatar_url is None


# ---------------------------------------------------------------------------
# Allowed-domain matching
# ---------------------------------------------------------------------------


def test_no_configured_domains_means_no_restriction():
    """The default. A customer who never asked for a restriction keeps working."""
    chatbot = make_chatbot(allowed_domains=[])
    deployment.check_embedding_origin(chatbot, embed_origin="https://anywhere.test")


def test_an_authorised_host_is_allowed():
    chatbot = make_chatbot(allowed_domains=["acme.test"])
    deployment.check_embedding_origin(chatbot, embed_origin="https://acme.test")


def test_an_unauthorised_host_is_refused():
    chatbot = make_chatbot(allowed_domains=["acme.test"])
    with pytest.raises(errors.AppError) as caught:
        deployment.check_embedding_origin(chatbot, embed_origin="https://evil.test")
    assert caught.value.code == "embedding_not_allowed"


def test_a_missing_origin_is_refused_when_a_list_is_configured():
    """We cannot tell where it came from, and a restriction was requested."""
    chatbot = make_chatbot(allowed_domains=["acme.test"])
    with pytest.raises(errors.AppError):
        deployment.check_embedding_origin(chatbot)


@pytest.mark.parametrize(
    ("host", "allowed"),
    [
        ("https://shop.acme.test", True),
        ("https://acme.test", True),
        ("https://deep.shop.acme.test", True),
        ("https://notacme.test", False),
        # The classic suffix-matching bug: a naive `endswith` accepts this.
        ("https://acme.test.evil.net", False),
        ("https://evilacme.test", False),
    ],
)
def test_a_wildcard_matches_subdomains_but_not_lookalikes(host: str, allowed: bool):
    chatbot = make_chatbot(allowed_domains=["*.acme.test"])
    if allowed:
        deployment.check_embedding_origin(chatbot, embed_origin=host)
    else:
        with pytest.raises(errors.AppError):
            deployment.check_embedding_origin(chatbot, embed_origin=host)


def test_our_own_host_is_always_allowed():
    """The standalone link and the dashboard preview are not third-party embeds."""
    chatbot = make_chatbot(allowed_domains=["acme.test"])
    own = get_settings().public_widget_base_url
    deployment.check_embedding_origin(chatbot, embed_origin=own)


def test_the_explicit_embed_origin_wins_over_the_request_origin():
    """Inside an iframe the browser's Origin is *ours*, not the host page's.

    So the check must prefer the forwarded host-page origin, or a restricted
    chatbot would accept every embed simply because the iframe is ours.
    """
    chatbot = make_chatbot(allowed_domains=["acme.test"])
    with pytest.raises(errors.AppError):
        deployment.check_embedding_origin(
            chatbot,
            embed_origin="https://evil.test",
            origin="https://acme.test",
        )


def test_the_referer_is_used_when_nothing_better_exists():
    chatbot = make_chatbot(allowed_domains=["acme.test"])
    deployment.check_embedding_origin(
        chatbot, referer="https://acme.test/contact?utm=x"
    )


def test_a_port_in_the_pattern_must_match():
    chatbot = make_chatbot(allowed_domains=["localhost:5500"])
    deployment.check_embedding_origin(chatbot, embed_origin="http://localhost:5500")
    with pytest.raises(errors.AppError):
        deployment.check_embedding_origin(chatbot, embed_origin="http://localhost:9999")


def test_a_pattern_without_a_port_ignores_the_requests_port():
    chatbot = make_chatbot(allowed_domains=["acme.test"])
    deployment.check_embedding_origin(chatbot, embed_origin="https://acme.test:8443")


def test_the_standalone_link_is_not_refused_by_a_configured_domain_list():
    """The shareable link and the dashboard preview must keep working.

    Both are served from our own host, so the embed page sends no
    `X-Embed-Origin` (it is not in an iframe on someone else's page) and the
    browser's `Origin` is ours. A restricted chatbot must still answer: the
    customer restricted *where their widget may be embedded*, not whether
    they can open their own chatbot.
    """
    chatbot = make_chatbot(allowed_domains=["acme.test"])
    own = get_settings().public_widget_base_url
    deployment.check_embedding_origin(chatbot, origin=own)


def test_the_dashboard_preview_iframe_is_allowed():
    """The Deploy tab iframes the real panel URL from the dashboard origin."""
    chatbot = make_chatbot(allowed_domains=["acme.test"])
    own = get_settings().public_widget_base_url
    deployment.check_embedding_origin(chatbot, embed_origin=own, origin=own)


# ---------------------------------------------------------------------------
# Publication gate configuration
# ---------------------------------------------------------------------------


def test_the_approval_allowlist_is_parsed_case_insensitively():
    original = get_settings()
    settings = Settings(
        database_url=original.database_url,
        firecrawl_api_key=original.firecrawl_api_key,
        publish_allowed_emails="Owner@Acme.TEST, second@acme.test ,",
    )
    assert settings.publish_allowed_email_set == {
        "owner@acme.test",
        "second@acme.test",
    }


def test_an_empty_allowlist_parses_to_an_empty_set():
    """So the gate refuses everyone rather than accidentally allowing all."""
    original = get_settings()
    settings = Settings(
        database_url=original.database_url,
        firecrawl_api_key=original.firecrawl_api_key,
        publish_allowed_emails="",
    )
    assert settings.publish_allowed_email_set == set()

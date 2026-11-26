"""The generated loader script and the install snippets.

Two kinds of assertion here:

  * **It is valid JavaScript.** The loader is built by string substitution and
    served to customers' production websites. A syntax error would be silent
    on our side and visible on theirs, so the script is parsed with Node when
    Node is available.
  * **It contains what it should and nothing it should not.** The loader runs
    inside the customer's page with full access to their DOM, so anything
    secret or internal appearing in it is a disclosure.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import uuid

import pytest

from config import Settings, get_settings
from helpers import deployment, widget
from models.base import ChatbotStatus
from models.chatbot import Chatbot

PUBLIC_ID = "abc234defghijkmnpqrstu"


def make_chatbot(**overrides) -> Chatbot:
    defaults = dict(
        id=uuid.uuid4(),
        public_id=PUBLIC_ID,
        owner_id=uuid.uuid4(),
        name="Acme Interiors",
        website_url="https://acme.test/",
        status=ChatbotStatus.PUBLISHED.value,
        crawl_status="completed",
        draft_config={},
        published_config={"ai": {}, "deployment": {}},
        deployment_config={},
        allowed_domains=[],
    )
    defaults.update(overrides)
    return Chatbot(**defaults)


def build(chatbot: Chatbot | None = None) -> str:
    bot = chatbot or make_chatbot()
    return widget.build_loader_script(bot, deployment.resolve_launcher(bot, published=True))


# ---------------------------------------------------------------------------
# It parses
# ---------------------------------------------------------------------------

NODE = shutil.which("node")

# Parse the script without executing it and without a temporary file: the
# Function constructor compiles its body and throws SyntaxError on bad input,
# but never runs it, so no DOM is needed.
_PARSE_PROGRAM = (
    "let s='';"
    "process.stdin.setEncoding('utf8');"
    "process.stdin.on('data', d => s += d);"
    "process.stdin.on('end', () => {"
    "  try { new Function(s); }"
    "  catch (e) { console.error(String(e)); process.exit(1); }"
    "});"
)


def assert_parses(source: str) -> None:
    """Fail with Node's own syntax error when `source` is not valid JavaScript."""
    if NODE is None:
        pytest.skip("Node is not installed")
    result = subprocess.run(
        [NODE, "-e", _PARSE_PROGRAM],
        input=source,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.strip()


def test_the_generated_loader_is_valid_javascript():
    assert_parses(build())


def test_the_unavailable_loader_is_valid_javascript():
    assert_parses(widget.build_unavailable_script(PUBLIC_ID))


def test_a_hostile_chatbot_name_cannot_break_out_of_the_script():
    """Every embedded value goes through json.dumps for exactly this reason."""
    assert_parses(
        build(make_chatbot(name='</script><script>alert("xss")//" + \n `${x}` \\ \''))
    )


def test_a_hostile_launcher_icon_cannot_break_out_of_the_script():
    """`icon_url` is interpolated into an HTML string inside the script."""
    assert_parses(
        build(
            make_chatbot(
                published_config={
                    "ai": {},
                    "deployment": {
                        "icon_url": 'https://a.test/x.png" onerror="alert(1)'
                    },
                }
            )
        )
    )


def test_the_configuration_block_is_valid_json():
    """The substituted config must be parseable, not merely look right."""
    script = build()
    start = script.index("var C = ") + len("var C = ")
    end = script.index(";\n", start)
    config = json.loads(script[start:end])
    assert config["id"] == PUBLIC_ID


# ---------------------------------------------------------------------------
# What it contains
# ---------------------------------------------------------------------------


def test_the_loader_points_the_iframe_at_the_panel_url():
    settings = get_settings()
    script = build()
    assert settings.embed_url(PUBLIC_ID, mode="panel") in script


def test_the_loader_carries_the_launcher_appearance():
    chatbot = make_chatbot(
        published_config={
            "ai": {},
            "deployment": {
                "launcher_size": 64,
                "position": "bottom-left",
                "accent_color": "#1f5a8a",
            },
        }
    )
    script = build(chatbot)
    assert '"size": 64' in script or '"size":64' in script
    assert '"side": "left"' in script or '"side":"left"' in script
    assert "#1f5a8a" in script


def test_the_loader_addresses_postmessage_to_an_explicit_origin():
    """A `"*"` target would broadcast to whatever happens to be in the frame."""
    script = build()
    assert "postMessage(message, C.origin)" in script
    assert 'postMessage(message, "*")' not in script


def test_the_loader_validates_inbound_message_origin_and_sender():
    script = build()
    assert "event.origin !== C.origin" in script
    assert "event.source !== iframe.contentWindow" in script
    assert 'data.source !== "skanda-ai-widget"' in script


def test_the_collapsed_widget_does_not_cover_the_host_page():
    """The container is zero-sized and click-through.

    A transparent full-viewport iframe is the common implementation of this
    feature and it swallows every click on the customer's site.
    """
    script = build()
    assert 'pointerEvents: "none"' in script
    assert 'width: "0", height: "0"' in script


def test_the_iframe_sandbox_keeps_the_frames_own_origin():
    """`allow-same-origin` must be present, and is not a weakening.

    The frame is served from our domain, so it is already cross-origin to the
    customer's site and the browser blocks DOM access on that basis. Omitting
    the token does not add protection — it gives the frame an *opaque* origin,
    which breaks the API calls (`Origin: null` fails our CORS allow-list),
    localStorage, and the postMessage origin checks on both sides.

    This test exists because omitting it is the intuitive-looking mistake, and
    it produces a widget that silently never works.
    """
    script = build()
    marker = '"sandbox",'
    value = script.split(marker, 1)[1].split('"', 2)[1]
    tokens = set(value.split())
    assert "allow-same-origin" in tokens
    assert "allow-scripts" in tokens
    # Still sandboxed: the frame cannot navigate the customer's page away.
    assert "allow-top-navigation" not in tokens


def test_the_loader_handles_mobile_without_horizontal_overflow():
    script = build()
    assert "isMobile" in script
    assert f'"mobile": {widget.MOBILE_BREAKPOINT}' in script or (
        f'"mobile":{widget.MOBILE_BREAKPOINT}' in script
    )


def test_a_resize_request_from_the_frame_is_clamped():
    """A compromised frame must not be able to cover the customer's page."""
    script = build()
    assert "Math.min(Math.max(h, 320), C.h)" in script


def test_the_loader_guards_against_being_installed_twice():
    """And claims the marker immediately, not on DOMContentLoaded.

    Two script tags both evaluate before DOMContentLoaded, so a guard that
    waited for the public handle to be published would let both through.
    """
    script = build()
    body = script.split("var C = ", 1)[1]
    guard = body.index("__skandaAILoaded")
    build_call = body.index("function build()")
    assert guard < build_call, "the duplicate guard must run before any setup"
    assert "window.__skandaAILoaded[C.id] = true" in script


# ---------------------------------------------------------------------------
# What it must NOT contain
# ---------------------------------------------------------------------------


def test_the_loader_contains_no_internal_identifiers():
    """It runs in the customer's page. Only the public handle belongs there."""
    chatbot = make_chatbot()
    script = build(chatbot)
    assert str(chatbot.id) not in script
    assert str(chatbot.owner_id) not in script


def test_the_loader_contains_no_credentials_or_private_configuration():
    chatbot = make_chatbot(
        published_config={
            "ai": {
                "persona": "Internal persona note",
                "instructions": "Never mention refunds",
                "fallback_message": "Call the private line",
            },
            "deployment": {},
        }
    )
    script = build(chatbot)
    for secret in (
        "Internal persona",
        "Never mention refunds",
        "private line",
        "gemma",
        "GEMMA",
        "firecrawl",
        "supabase",
        "postgres",
    ):
        assert secret not in script, f"the loader leaks {secret!r}"


def test_the_loader_contains_no_ai_or_retrieval_logic():
    """Iframe-first: the loader positions a frame, nothing more."""
    script = build()
    for term in ("retrieval", "knowledge", "prompt", "generateContent", "completions"):
        assert term not in script.lower()


def test_the_unavailable_script_does_not_confirm_whether_the_handle_exists():
    body = widget.build_unavailable_script("doesnotexist")
    assert "not published" in body
    # Identical wording regardless of the reason, so the response does not
    # distinguish "unknown" from "unpublished".
    assert "unknown" not in body.lower()


# ---------------------------------------------------------------------------
# Size
# ---------------------------------------------------------------------------


def test_the_loader_is_small():
    """A budget, not a target hit exactly: ~4.5 KB gzipped.

    The brief named 2-3 KB compressed as a design goal and said explicitly
    that it should not compromise reliability. Phase 4 spent some of it: the
    launcher gained shapes, an optional label, four opening animations and
    four attention animations, which means a keyframes stylesheet and the
    `prefers-reduced-motion` check that stops them. That is behaviour the
    customer chose, not padding.

    The comments are the rest. They are kept deliberately — this file runs on
    the customer's own website and their developer may well read it, and an
    unexplained loader is harder to trust than a slightly larger one. The
    ceiling below is a tripwire against drift, not an optimisation target.
    """
    import gzip

    raw = build().encode("utf-8")
    compressed = len(gzip.compress(raw))
    assert len(raw) < 14_000, f"loader is {len(raw)} bytes uncompressed"
    assert compressed < 5_000, f"loader is {compressed} bytes gzipped"


# ---------------------------------------------------------------------------
# Install snippets
# ---------------------------------------------------------------------------


def snippets(chatbot: Chatbot | None = None):
    bot = chatbot or make_chatbot()
    return {
        snippet.mode: snippet
        for snippet in widget.build_snippets(
            bot, deployment.resolve_launcher(bot, published=False)
        )
    }


def test_three_deployment_modes_are_offered():
    assert set(snippets()) == {"widget", "iframe", "link"}


def test_the_floating_widget_is_the_recommended_option():
    assert snippets()["widget"].recommended is True


def test_the_widget_snippet_is_a_single_async_script_tag():
    code = snippets()["widget"].code or ""
    assert code.count("<script") == 1
    assert "async" in code
    assert get_settings().widget_script_url(PUBLIC_ID) in code


def test_the_iframe_snippet_matches_the_brief():
    code = snippets()["iframe"].code or ""
    assert code.startswith("<iframe")
    assert 'width="100%"' in code
    assert 'style="border:0;"' in code
    assert 'loading="lazy"' in code
    assert "title=" in code
    assert get_settings().embed_url(PUBLIC_ID) in code


def test_the_standalone_link_has_no_code_to_paste():
    link = snippets()["link"]
    assert link.code is None
    assert link.url.endswith("?mode=full")


def test_a_hostile_chatbot_name_is_escaped_in_the_iframe_snippet():
    """The snippet is shown in the dashboard and pasted into a real page."""
    code = snippets(make_chatbot(name='Acme" onload="alert(1)'))["iframe"].code
    assert 'onload="alert(1)' not in (code or "")
    assert "&quot;" in (code or "")


def test_no_url_contains_a_hardcoded_production_domain():
    """Every URL is generated from PUBLIC_WIDGET_BASE_URL."""
    for snippet in snippets().values():
        assert "chat.skandaai.in" not in snippet.url
        assert "chat.skandaai.in" not in (snippet.code or "")


def test_urls_follow_the_configured_deployment_domain(monkeypatch: pytest.MonkeyPatch):
    original = get_settings()
    moved = Settings(
        database_url=original.database_url,
        firecrawl_api_key=original.firecrawl_api_key,
        public_widget_base_url="https://chat.example.com/",
    )
    monkeypatch.setattr(widget, "get_settings", lambda: moved)

    chatbot = make_chatbot()
    built = widget.build_snippets(
        chatbot, deployment.resolve_launcher(chatbot, published=False)
    )
    for snippet in built:
        assert snippet.url.startswith("https://chat.example.com/embed/")
    # The trailing slash is normalised, not doubled.
    assert "//embed" not in built[0].url


# ---------------------------------------------------------------------------
# Platform instructions
# ---------------------------------------------------------------------------


def test_the_required_platforms_are_covered():
    names = " ".join(p.platform for p in widget.platform_instructions()).lower()
    for platform in ("html", "wordpress", "shopify", "react"):
        assert platform in names


def test_platforms_that_block_custom_scripts_are_flagged():
    """The brief: "Clearly state when a website platform restricts JavaScript"."""
    restricted = [
        p for p in widget.platform_instructions() if not p.supports_javascript
    ]
    assert restricted, "no platform is flagged as restricting custom JavaScript"
    assert all(p.note for p in restricted)


# ---------------------------------------------------------------------------
# Phase 4 — the launcher the customisation studio describes
# ---------------------------------------------------------------------------


def launcher_config(**deployment_settings) -> dict:
    """The config block the loader is built with, for one launcher setup."""
    bot = make_chatbot(
        published_config={"ai": {}, "deployment": deployment_settings}
    )
    source = build(bot)
    block = source.split("var C = ", 1)[1].split(";\n", 1)[0]
    return json.loads(block)


@pytest.mark.parametrize(
    ("shape", "expected"),
    [("circle", "50%"), ("square", "10px"), ("rounded", "18px")],
)
def test_each_launcher_shape_reaches_the_loader_as_a_radius(shape: str, expected: str):
    """The shape is resolved to a CSS length server-side, so "circle" stays
    circular at every size the customer can choose."""
    assert launcher_config(shape=shape, launcher_size=56)["radius"] == expected


def test_a_circle_stays_circular_at_the_largest_allowed_size():
    assert launcher_config(shape="circle", launcher_size=96)["radius"] == "50%"


def test_a_rounded_launcher_scales_its_radius_with_its_size():
    small = launcher_config(shape="rounded", launcher_size=40)["radius"]
    large = launcher_config(shape="rounded", launcher_size=96)["radius"]
    assert int(small.removesuffix("px")) < int(large.removesuffix("px"))


def test_the_launcher_label_is_carried_and_set_as_text_not_markup():
    """It is the customer's own text going into their own page; as markup it
    would be an injection point on their website."""
    source = build(
        make_chatbot(
            published_config={"ai": {}, "deployment": {"label": "Chat with us"}}
        )
    )
    assert '"label": "Chat with us"' in source
    assert "span.textContent = C.label" in source
    assert "innerHTML = C.label" not in source


def test_a_hostile_launcher_label_cannot_break_out_of_the_script():
    assert_parses(
        build(
            make_chatbot(
                published_config={
                    "ai": {},
                    "deployment": {"label": '</script><img src=x onerror=alert(1)>'},
                }
            )
        )
    )


@pytest.mark.parametrize("animation", ["scale", "slide", "fade", "none"])
def test_every_opening_animation_is_honoured(animation: str):
    config = launcher_config(open_animation=animation)
    assert config["anim"] == animation


@pytest.mark.parametrize("animation", ["pulse", "bounce", "glow", "none"])
def test_every_attention_animation_is_honoured(animation: str):
    assert launcher_config(attention_animation=animation)["attn"] == animation


def test_the_keyframes_for_every_offered_animation_exist_in_the_loader():
    """An animation named in the studio but missing from the stylesheet would
    simply do nothing, which is the kind of dead control this phase is meant
    to eliminate."""
    source = build()
    for name in ("skandaAiScale", "skandaAiSlide", "skandaAiFade"):
        assert f"@keyframes {name}" in source
    for name in ("skandaAiPulse", "skandaAiBounce", "skandaAiGlow"):
        assert f"@keyframes {name}" in source


def test_animations_are_skipped_for_a_visitor_who_asked_for_less_motion():
    source = build()
    assert "prefers-reduced-motion: reduce" in source
    assert "reducedMotion()" in source
    # Both the opening animation and the attention loop consult it.
    assert source.count("reducedMotion()") >= 3


def test_the_keyframe_stylesheet_is_injected_only_once_per_page():
    """Two chatbots on one page must not fight over a stylesheet."""
    source = build()
    assert 'doc.getElementById("skanda-ai-keyframes")' in source
    assert 'el.id = "skanda-ai-keyframes"' in source


def test_the_window_corner_radius_comes_from_the_appearance_settings():
    bot = make_chatbot()
    source = widget.build_loader_script(
        bot, deployment.resolve_launcher(bot, published=True), window_radius=6
    )
    assert '"wr": 6' in source


def test_an_absurd_window_radius_cannot_produce_a_pill_shaped_window():
    bot = make_chatbot()
    source = widget.build_loader_script(
        bot, deployment.resolve_launcher(bot, published=True), window_radius=9_000
    )
    assert '"wr": 28' in source


def test_the_loader_still_carries_no_private_configuration():
    """Phase 4 added settings; none of the private ones may travel.

    The loader runs inside the customer's page, so everything in it is
    readable by anyone who can read their website's source.
    """
    bot = make_chatbot(
        published_config={
            "ai": {
                "persona": "Terse and slightly rude, internal note",
                "instructions": "Always upsell the premium package",
                "fallback_message": "Ring the private line",
                "business_name": "Acme Interiors",
            },
            "appearance": {"colors": {"primary": "#0f766e"}},
            "deployment": {"label": "Chat"},
        }
    )
    source = build(bot)
    for secret in (
        "slightly rude",
        "upsell",
        "private line",
        str(bot.id),
        str(bot.owner_id),
        "gemma",
        "Gemma",
    ):
        assert secret not in source, f"the loader leaks {secret!r}"

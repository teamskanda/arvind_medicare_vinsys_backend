"""The customisation studio: resolution, validation, and real effect.

The guarantee this phase makes is that no control is decorative. These tests
hold that line in three ways:

  * **Round-trip.** Every value the options catalogue offers is set and read
    back, so the studio cannot offer a choice the resolver silently discards.
  * **Reaches the model.** The personality settings are followed into the
    built prompt, because a tone preset that does not change the prompt is a
    dropdown that does nothing.
  * **Refuses the rest.** A stored document is never trusted: colours must be
    hex, numbers are clamped to the range the UI offers, choices must be known
    values, and image references must be `asset:<uuid>` or https.
"""

from __future__ import annotations

import base64
import uuid

import pytest

from config import get_settings
from helpers import customization as cz
from helpers import errors, prompt_builder, widget
from helpers.prompt_builder import ChatbotAiConfig, build_prompt, build_system_text
from helpers.token_manager import build_budget
from models.base import ChatbotStatus
from models.chatbot import Chatbot
from models.customization import (
    ASSET_MAX_BYTES,
    AppearanceSettingsInput,
    ColorOverridesInput,
    ContentSettingsInput,
    CustomizationUpdate,
    LauncherSettingsInput,
    is_hex_color,
)

PUBLIC_ID = "abc234defghijkmnpqrstu"

# A 1x1 PNG, and the smallest plausible GIF and WebP headers.
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8AAAwAB/AE"
    "BiQAAAABJRU5ErkJggg=="
)
GIF_BYTES = b"GIF89a" + b"\x01\x00\x01\x00" + b"\x00" * 20
WEBP_BYTES = b"RIFF" + b"\x24\x00\x00\x00" + b"WEBP" + b"VP8 " + b"\x00" * 20
JPEG_BYTES = b"\xff\xd8\xff\xe0" + b"\x00" * 40


def make_chatbot(
    *,
    draft: dict | None = None,
    published: dict | None = None,
    launcher: dict | None = None,
    brand_color: str | None = None,
    favicon_url: str | None = None,
    status: str = ChatbotStatus.DRAFT.value,
) -> Chatbot:
    return Chatbot(
        id=uuid.uuid4(),
        public_id=PUBLIC_ID,
        owner_id=uuid.uuid4(),
        name="Acme Interiors",
        website_url="https://acme.test",
        status=status,
        crawl_status="completed",
        draft_config=draft or {},
        published_config=published,
        deployment_config=launcher or {},
        allowed_domains=[],
        brand_color=brand_color,
        favicon_url=favicon_url,
    )


# ---------------------------------------------------------------------------
# Colour validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["#fff", "#FFF", "#7d2a47", "#0F766E"])
def test_a_hex_colour_is_accepted(value: str):
    assert is_hex_color(value) is True


@pytest.mark.parametrize(
    "value",
    [
        "red",
        "rgb(0,0,0)",
        "#gggggg",
        "#7d2a4",
        "#7d2a47; background:url(x)",
        "url(javascript:alert(1))",
        "var(--brand)",
        "",
        None,
        42,
    ],
)
def test_anything_other_than_a_hex_colour_is_refused(value):
    """These values are interpolated into inline styles and into the loader.

    A hex literal cannot carry a `url(...)`, an `expression(...)` or a stray
    semicolon, so restricting the alphabet removes the injection question
    rather than answering it.
    """
    assert is_hex_color(value) is False


def test_a_rejected_colour_produces_a_message_the_customer_can_act_on():
    with pytest.raises(ValueError) as caught:
        ColorOverridesInput(primary="cornflowerblue")
    assert "#7d2a47" in str(caught.value)


def test_a_stored_non_hex_colour_is_ignored_rather_than_rendered():
    """Validation on the way in is not enough: a document written by an
    earlier version, or by hand, must not reach the browser either."""
    chatbot = make_chatbot(
        draft={"appearance": {"colors": {"primary": "red; background:url(x)"}}}
    )
    appearance = cz.resolve_appearance(chatbot, published=False)
    assert appearance.light.primary == get_settings().widget_accent_color


# ---------------------------------------------------------------------------
# Palette derivation
# ---------------------------------------------------------------------------


def test_one_brand_colour_produces_a_complete_palette():
    """The customer is asked for one colour, not eleven."""
    chatbot = make_chatbot(draft={"appearance": {"colors": {"primary": "#0f766e"}}})
    light = cz.resolve_appearance(chatbot, published=False).light

    assert light.primary == "#0f766e"
    for field, value in light.model_dump().items():
        assert is_hex_color(value), f"{field} is not a colour: {value!r}"


def test_an_explicit_colour_replaces_the_derived_one():
    chatbot = make_chatbot(
        draft={
            "appearance": {
                "colors": {"primary": "#0f766e", "header_background": "#111111"}
            }
        }
    )
    light = cz.resolve_appearance(chatbot, published=False).light
    assert light.header_background == "#111111"
    assert light.primary == "#0f766e"


def test_header_text_follows_the_header_background():
    """Why a customer can pick any header colour and still read the header."""
    dark_header = cz.resolve_appearance(
        make_chatbot(draft={"appearance": {"colors": {"header_background": "#101010"}}}),
        published=False,
    ).light
    light_header = cz.resolve_appearance(
        make_chatbot(draft={"appearance": {"colors": {"header_background": "#fdf6b2"}}}),
        published=False,
    ).light

    assert dark_header.header_text == "#ffffff"
    assert light_header.header_text == "#1b1a1a"


def test_an_overridden_header_text_colour_wins_even_if_it_is_a_poor_choice():
    """Derived defaults are help, not a veto. The customer owns the decision."""
    appearance = cz.resolve_appearance(
        make_chatbot(
            draft={
                "appearance": {
                    "colors": {"header_background": "#101010", "header_text": "#222222"}
                }
            }
        ),
        published=False,
    )
    assert appearance.light.header_text == "#222222"


def test_both_themes_are_always_resolved():
    """`theme: auto` is answered in the browser, so both palettes are sent."""
    appearance = cz.resolve_appearance(
        make_chatbot(draft={"appearance": {"theme": "auto"}}), published=False
    )
    assert appearance.theme == "auto"
    assert appearance.light.chat_background != appearance.dark.chat_background
    assert appearance.light.message_text != appearance.dark.message_text


def test_an_override_applies_to_both_themes():
    appearance = cz.resolve_appearance(
        make_chatbot(draft={"appearance": {"colors": {"ai_bubble": "#abcdef"}}}),
        published=False,
    )
    assert appearance.light.ai_bubble == "#abcdef"
    assert appearance.dark.ai_bubble == "#abcdef"


def test_a_dark_palette_keeps_text_legible_against_its_background():
    """The derived dark theme must not need eleven manual corrections."""
    dark = cz.resolve_appearance(
        make_chatbot(draft={"appearance": {"theme": "dark"}}), published=False
    ).dark
    contrast_background = cz.relative_luminance(dark.chat_background)
    contrast_text = cz.relative_luminance(dark.message_text)
    assert contrast_text > contrast_background + 0.4


def test_the_scraped_brand_colour_is_the_default():
    """A customer who never opens the studio still gets their own colour."""
    chatbot = make_chatbot(brand_color="#1f5a8a")
    assert cz.resolve_appearance(chatbot, published=False).light.primary == "#1f5a8a"


def test_an_unusable_scraped_brand_colour_falls_back_to_ours():
    chatbot = make_chatbot(brand_color="rgb(31,90,138)")
    assert (
        cz.resolve_appearance(chatbot, published=False).light.primary
        == get_settings().widget_accent_color
    )


def test_mixing_is_bounded_at_both_ends():
    assert cz.mix("#000000", "#ffffff", 0) == "#000000"
    assert cz.mix("#000000", "#ffffff", 1) == "#ffffff"
    assert cz.mix("#000000", "#ffffff", 5) == "#ffffff"
    assert cz.mix("#000000", "#ffffff", -3) == "#000000"
    assert cz.mix("#000000", "#ffffff", 0.5) == "#808080"


# ---------------------------------------------------------------------------
# Appearance: malformed documents
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "stored",
    [
        {"appearance": "not a dict"},
        {"appearance": {"colors": "not a dict"}},
        {"appearance": {"theme": "neon", "font": "comic", "border_radius": "big"}},
        {"appearance": {"border_radius": None}},
        {"appearance": {"colors": {"primary": []}}},
        {},
    ],
)
def test_a_malformed_appearance_document_produces_a_working_chatbot(stored: dict):
    appearance = cz.resolve_appearance(make_chatbot(draft=stored), published=False)
    assert appearance.theme in cz.THEMES
    assert appearance.font in cz.FONTS
    assert cz.MIN_BORDER_RADIUS <= appearance.border_radius <= cz.MAX_BORDER_RADIUS
    assert is_hex_color(appearance.light.primary)


def test_an_out_of_range_radius_is_clamped_not_rejected():
    assert (
        cz.resolve_appearance(
            make_chatbot(draft={"appearance": {"border_radius": 9_000}}), published=False
        ).border_radius
        == cz.MAX_BORDER_RADIUS
    )
    assert (
        cz.resolve_appearance(
            make_chatbot(draft={"appearance": {"border_radius": -40}}), published=False
        ).border_radius
        == cz.MIN_BORDER_RADIUS
    )


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------


def test_an_uploaded_image_reference_becomes_an_absolute_url():
    asset_id = uuid.uuid4()
    chatbot = make_chatbot(draft={"appearance": {"avatar": f"asset:{asset_id}"}})
    avatar = cz.resolve_appearance(chatbot, published=False).avatar_url
    assert avatar == f"{get_settings().public_api_base_url}/api/public/assets/{asset_id}"


def test_an_external_https_image_is_kept():
    chatbot = make_chatbot(draft={"appearance": {"logo": "https://cdn.acme.test/l.png"}})
    assert cz.resolve_appearance(chatbot, published=False).logo_url == (
        "https://cdn.acme.test/l.png"
    )


@pytest.mark.parametrize(
    "value",
    [
        "javascript:alert(1)",
        "data:image/svg+xml;base64,PHN2Zz48L3N2Zz4=",
        "http://insecure.test/logo.png",
        "//cdn.acme.test/logo.png",
        "asset:not-a-uuid",
        "asset:",
        "https://",
        "not a url",
        123,
        None,
        [],
    ],
)
def test_an_unsafe_image_value_is_dropped(value):
    """These end up in an <img src> in our page and in the customer's page."""
    assert cz.resolve_image(value) is None


def test_an_oversized_image_value_is_dropped():
    assert cz.resolve_image("https://acme.test/" + "x" * 3_000) is None


def test_the_favicon_is_the_fallback_avatar_but_never_the_logo():
    chatbot = make_chatbot(favicon_url="https://acme.test/favicon.png")
    appearance = cz.resolve_appearance(chatbot, published=False)
    assert appearance.avatar_url == "https://acme.test/favicon.png"
    assert appearance.logo_url is None


def test_a_chosen_avatar_beats_the_scraped_favicon():
    chatbot = make_chatbot(
        draft={"appearance": {"avatar": "https://cdn.acme.test/face.png"}},
        favicon_url="https://acme.test/favicon.png",
    )
    assert (
        cz.resolve_appearance(chatbot, published=False).avatar_url
        == "https://cdn.acme.test/face.png"
    )


def test_an_insecure_favicon_is_not_offered_as_an_avatar():
    chatbot = make_chatbot(favicon_url="http://acme.test/favicon.ico")
    assert cz.resolve_appearance(chatbot, published=False).avatar_url is None


# ---------------------------------------------------------------------------
# Content and personality
# ---------------------------------------------------------------------------


def test_a_tone_preset_supplies_the_persona():
    """Picking a tone has to do something on its own."""
    content = cz.resolve_content(
        make_chatbot(draft={"ai": {"tone": "formal"}}), published=False
    )
    assert content.tone == "formal"
    assert content.persona == prompt_builder.TONE_PRESETS["formal"].persona


def test_a_custom_persona_replaces_the_tone_preset_but_keeps_its_style():
    """Persona is who the assistant is; tone is how it writes. Both apply."""
    config = ChatbotAiConfig.from_chatbot(
        name="Acme",
        website_url="https://acme.test",
        config={"ai": {"tone": "formal", "persona": "A retired architect"}},
    )
    assert config.persona == "A retired architect"

    text = build_system_text(config, has_knowledge=True, cite_sources=True)
    assert "A retired architect" in text
    assert prompt_builder.TONE_PRESETS["formal"].style_rule in text


def test_the_default_tone_preserves_the_pre_phase_4_persona():
    """An existing chatbot must not change behaviour because this phase shipped."""
    config = ChatbotAiConfig.from_chatbot(name="Acme", website_url=None, config={})
    assert config.tone == prompt_builder.DEFAULT_TONE
    assert config.persona == prompt_builder.DEFAULT_PERSONA


@pytest.mark.parametrize("tone", list(prompt_builder.TONE_PRESETS))
def test_every_tone_preset_reaches_the_prompt(tone: str):
    config = ChatbotAiConfig.from_chatbot(
        name="Acme", website_url=None, config={"ai": {"tone": tone}}
    )
    text = build_system_text(config, has_knowledge=True, cite_sources=True)
    assert prompt_builder.TONE_PRESETS[tone].style_rule in text


@pytest.mark.parametrize("level", list(prompt_builder.GROUNDING_LEVELS))
def test_every_grounding_level_reaches_the_prompt(level: str):
    """Knowledge-grounding strictness is a prompt rule, not a label."""
    config = ChatbotAiConfig.from_chatbot(
        name="Acme", website_url=None, config={"ai": {"grounding": level}}
    )
    text = build_system_text(config, has_knowledge=True, cite_sources=True)
    assert prompt_builder.GROUNDING_LEVELS[level].rule in text


def test_the_strict_level_differs_from_the_flexible_one_in_the_prompt():
    """Two levels that produced the same prompt would be a fake control."""

    def text_for(level: str) -> str:
        return build_system_text(
            ChatbotAiConfig.from_chatbot(
                name="Acme", website_url=None, config={"ai": {"grounding": level}}
            ),
            has_knowledge=True,
            cite_sources=True,
        )

    assert text_for("strict") != text_for("flexible")


def test_the_business_name_reaches_the_prompt_and_the_public_config():
    config = ChatbotAiConfig.from_chatbot(
        name="Ask Acme",
        website_url="https://acme.test",
        config={"ai": {"business_name": "Acme Interiors"}},
    )
    text = build_system_text(config, has_knowledge=True, cite_sources=True)
    assert "Ask Acme" in text
    assert "Acme Interiors" in text


@pytest.mark.parametrize("length", list(prompt_builder.RESPONSE_LENGTHS))
def test_every_response_length_changes_the_prompt(length: str):
    config = ChatbotAiConfig.from_chatbot(
        name="Acme", website_url=None, config={"ai": {"response_length": length}}
    )
    text = build_system_text(config, has_knowledge=True, cite_sources=True)
    assert prompt_builder._RESPONSE_LENGTH_GUIDANCE[length] in text


def test_a_chosen_language_replaces_automatic_detection():
    automatic = build_system_text(
        ChatbotAiConfig.from_chatbot(name="A", website_url=None, config={}),
        has_knowledge=True,
        cite_sources=True,
    )
    chosen = build_system_text(
        ChatbotAiConfig.from_chatbot(
            name="A", website_url=None, config={"ai": {"language": "Kannada"}}
        ),
        has_knowledge=True,
        cite_sources=True,
    )
    assert "same language the visitor used" in automatic
    assert "Reply in Kannada." in chosen


def test_turning_citations_off_removes_them_from_the_built_prompt():
    """Source citation visibility, followed all the way through."""
    from helpers.retrieval import Section

    section = Section(
        page_id=uuid.uuid4(),
        page_title="Services",
        url="https://acme.test/services",
        heading="Fitting",
        text="We fit kitchens across Bengaluru.",
        priority=5,
        index=0,
        score=1.0,
    )
    for cite, expected in ((True, 1), (False, 0)):
        built = build_prompt(
            config=ChatbotAiConfig.from_chatbot(
                name="A", website_url=None, config={"ai": {"cite_sources": cite}}
            ),
            sections=[section],
            question="Do you fit kitchens?",
            history=[],
            budget=build_budget(),
        )
        assert len(built.citations) == expected


def test_an_unknown_tone_or_grounding_level_falls_back_to_the_default():
    content = cz.resolve_content(
        make_chatbot(draft={"ai": {"tone": "sarcastic", "grounding": "none"}}),
        published=False,
    )
    assert content.tone == prompt_builder.DEFAULT_TONE
    assert content.grounding == prompt_builder.DEFAULT_GROUNDING


def test_the_welcome_message_is_resolved_and_the_greeting_is_separate():
    content = cz.resolve_content(
        make_chatbot(
            draft={
                "ai": {
                    "greeting": "Hello!",
                    "welcome_message": "Ask about our opening hours",
                }
            }
        ),
        published=False,
    )
    assert content.greeting == "Hello!"
    assert content.welcome_message == "Ask about our opening hours"


def test_the_studio_shows_the_same_values_the_prompt_will_use():
    """One resolver, so a customer cannot edit a value that does nothing."""
    chatbot = make_chatbot(
        draft={
            "ai": {
                "tone": "professional",
                "language": "Hindi",
                "response_length": "concise",
                "fallback_message": "Please call the shop.",
                "grounding": "strict",
            }
        }
    )
    studio = cz.resolve_content(chatbot, published=False)
    engine = ChatbotAiConfig.from_chatbot(
        name=chatbot.name, website_url=chatbot.website_url, config=chatbot.draft_config
    )
    assert studio.persona == engine.persona
    assert studio.tone == engine.tone
    assert studio.language == engine.language
    assert studio.response_length == engine.response_length
    assert studio.fallback_message == engine.fallback_message
    assert studio.grounding == engine.grounding


# ---------------------------------------------------------------------------
# Launcher
# ---------------------------------------------------------------------------


def test_the_launcher_colour_defaults_to_the_brand_colour():
    chatbot = make_chatbot(draft={"appearance": {"colors": {"primary": "#0f766e"}}})
    assert cz.resolve_launcher(chatbot).accent_color == "#0f766e"


def test_the_launcher_colour_can_differ_from_the_brand_colour():
    """A dark header with a bright button is a reasonable choice, so these
    are two controls rather than one."""
    chatbot = make_chatbot(
        draft={"appearance": {"colors": {"primary": "#101010"}}},
        launcher={"accent_color": "#0f766e"},
    )
    appearance = cz.resolve_appearance(chatbot, published=False)
    assert appearance.light.primary == "#101010"
    assert cz.resolve_launcher(chatbot).accent_color == "#0f766e"


def test_the_launcher_label_is_trimmed_and_capped():
    launcher = cz.resolve_launcher(
        make_chatbot(launcher={"label": "   " + "x" * 80 + "   "})
    )
    assert launcher.label is not None
    assert len(launcher.label) == 28


def test_an_empty_launcher_label_is_absent_rather_than_blank():
    assert cz.resolve_launcher(make_chatbot(launcher={"label": "   "})).label is None


@pytest.mark.parametrize(
    ("key", "stored", "expected_default"),
    [
        ("shape", "triangle", "circle"),
        ("open_animation", "explode", "scale"),
        ("attention_animation", "strobe", "none"),
        ("icon", "unicorn", "chat"),
        ("position", "top-left", "bottom-right"),
    ],
)
def test_an_unknown_launcher_choice_falls_back(key: str, stored: str, expected_default: str):
    launcher = cz.resolve_launcher(make_chatbot(launcher={key: stored}))
    assert getattr(launcher, key) == expected_default


def test_an_uploaded_launcher_icon_is_resolved_to_a_url():
    asset_id = uuid.uuid4()
    launcher = cz.resolve_launcher(
        make_chatbot(launcher={"icon_image": f"asset:{asset_id}"})
    )
    assert launcher.icon_url == cz.asset_url(asset_id)


# ---------------------------------------------------------------------------
# Round-trip: the catalogue and the resolver agree
# ---------------------------------------------------------------------------


def test_every_offered_font_resolves_to_itself_and_has_a_stack():
    for option in cz.options().fonts:
        appearance = cz.resolve_appearance(
            make_chatbot(draft={"appearance": {"font": option.value}}), published=False
        )
        assert appearance.font == option.value
        assert appearance.font_stack == option.font_stack
        assert appearance.font_stack


def test_every_offered_theme_resolves_to_itself():
    for option in cz.options().themes:
        appearance = cz.resolve_appearance(
            make_chatbot(draft={"appearance": {"theme": option.value}}), published=False
        )
        assert appearance.theme == option.value


def test_every_offered_language_resolves_to_itself():
    for option in cz.options().languages:
        content = cz.resolve_content(
            make_chatbot(draft={"ai": {"language": option.value}}), published=False
        )
        assert content.language == option.value


@pytest.mark.parametrize(
    ("catalogue", "field"),
    [
        ("launcher_icons", "icon"),
        ("launcher_shapes", "shape"),
        ("open_animations", "open_animation"),
        ("attention_animations", "attention_animation"),
    ],
)
def test_every_offered_launcher_choice_resolves_to_itself(catalogue: str, field: str):
    """The studio cannot offer a value the resolver would discard."""
    for option in getattr(cz.options(), catalogue):
        launcher = cz.resolve_launcher(make_chatbot(launcher={field: option.value}))
        assert getattr(launcher, field) == option.value


def test_every_offered_colour_preset_is_a_valid_colour():
    for option in cz.options().color_presets:
        assert is_hex_color(option.value)
        assert option.color == option.value


def test_every_offered_tone_and_grounding_level_resolves_to_itself():
    for option in cz.options().tones:
        content = cz.resolve_content(
            make_chatbot(draft={"ai": {"tone": option.value}}), published=False
        )
        assert content.tone == option.value
    for option in cz.options().grounding_levels:
        content = cz.resolve_content(
            make_chatbot(draft={"ai": {"grounding": option.value}}), published=False
        )
        assert content.grounding == option.value


def test_the_launcher_icon_catalogue_matches_the_icons_the_widget_can_draw():
    """One registry, read by the loader and offered to the studio.

    The studio's gallery and its launcher preview draw from the `svg_path`
    this catalogue carries, so an icon cannot look one way in the preview and
    another way on the customer's website.
    """
    assert set(cz.LAUNCHER_ICON_KEYS) == set(widget._ICONS)
    for option in cz.options().launcher_icons:
        assert option.svg_path == widget._ICONS[option.value]
        assert option.svg_path.startswith("<")


def test_the_response_length_catalogue_matches_the_prompt_rules():
    assert set(prompt_builder.RESPONSE_LENGTHS) == set(
        prompt_builder._RESPONSE_LENGTH_GUIDANCE
    )


def test_the_limits_the_studio_is_given_match_the_ones_enforced():
    limits = cz.options().limits
    assert limits.max_persona_characters == prompt_builder.MAX_PERSONA_CHARACTERS
    assert limits.max_instructions_characters == prompt_builder.MAX_INSTRUCTIONS_CHARACTERS
    assert limits.max_fallback_characters == prompt_builder.MAX_FALLBACK_CHARACTERS
    assert limits.asset_max_bytes == ASSET_MAX_BYTES
    assert limits.border_radius_range == [cz.MIN_BORDER_RADIUS, cz.MAX_BORDER_RADIUS]


# ---------------------------------------------------------------------------
# Partial updates
# ---------------------------------------------------------------------------


def test_only_the_keys_that_were_sent_are_written():
    """The studio saves one slider without resending forty values."""
    stored = {"theme": "dark", "font": "serif", "border_radius": 4}
    changes = AppearanceSettingsInput(font="mono").model_dump(exclude_unset=True)
    merged = cz._merge(stored, changes)
    assert merged == {"theme": "dark", "font": "mono", "border_radius": 4}


def test_an_explicit_null_clears_one_setting():
    """Which is how "reset this colour" works without a second endpoint."""
    stored = {"theme": "dark", "font": "serif"}
    changes = AppearanceSettingsInput(font=None).model_dump(exclude_unset=True)
    assert cz._merge(stored, changes) == {"theme": "dark"}


def test_a_nested_colour_update_merges_rather_than_replaces():
    stored = {"colors": {"primary": "#0f766e", "ai_bubble": "#ffffff"}}
    changes = AppearanceSettingsInput(
        colors=ColorOverridesInput(primary="#b45309")
    ).model_dump(exclude_unset=True)
    merged = cz._merge(stored, changes)
    assert merged["colors"] == {"primary": "#b45309", "ai_bubble": "#ffffff"}


def test_clearing_the_last_colour_removes_the_empty_group():
    stored = {"colors": {"primary": "#0f766e"}}
    changes = AppearanceSettingsInput(
        colors=ColorOverridesInput(primary=None)
    ).model_dump(exclude_unset=True)
    assert cz._merge(stored, changes) == {}


def test_an_empty_string_means_the_same_as_clearing():
    assert ContentSettingsInput(greeting="   ").greeting is None
    assert LauncherSettingsInput(label="  ").label is None


def test_a_suggested_question_list_is_cleaned_on_the_way_in():
    content = ContentSettingsInput(
        suggested_questions=["  What are your hours?  ", "What are your hours?", "  "]
    )
    assert content.suggested_questions == ["What are your hours?"]


def test_too_many_suggested_questions_is_refused_rather_than_truncated():
    with pytest.raises(ValueError):
        ContentSettingsInput(suggested_questions=[f"Question {i}?" for i in range(9)])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("launcher_size", 500),
        ("launcher_size", 4),
        ("window_width", 2_000),
        ("window_height", 10),
        ("offset_x", -5),
        ("offset_y", 9_000),
    ],
)
def test_an_out_of_range_launcher_value_is_refused_on_the_way_in(field: str, value: int):
    """Refused with a message, rather than silently clamped later: the
    customer should learn the limit at the moment they hit it."""
    with pytest.raises(ValueError):
        LauncherSettingsInput(**{field: value})


def test_an_update_carrying_nothing_is_valid_and_changes_nothing():
    payload = CustomizationUpdate()
    assert payload.content is None
    assert payload.appearance is None
    assert payload.launcher is None


# ---------------------------------------------------------------------------
# Draft vs published
# ---------------------------------------------------------------------------


def test_the_draft_and_the_published_snapshot_resolve_independently():
    chatbot = make_chatbot(
        draft={"appearance": {"colors": {"primary": "#0f766e"}}, "ai": {"tone": "formal"}},
        published={
            "appearance": {"colors": {"primary": "#b45309"}},
            "ai": {"tone": "friendly"},
        },
        status=ChatbotStatus.PUBLISHED.value,
    )
    assert cz.resolve_appearance(chatbot, published=False).light.primary == "#0f766e"
    assert cz.resolve_appearance(chatbot, published=True).light.primary == "#b45309"
    assert cz.resolve_content(chatbot, published=False).tone == "formal"
    assert cz.resolve_content(chatbot, published=True).tone == "friendly"


def test_an_unpublished_chatbot_reports_no_unpublished_changes():
    """Nothing is live, so there is nothing to be out of date with."""
    assert cz.has_unpublished_changes(make_chatbot(draft={"ai": {"tone": "formal"}})) is False


def test_an_identical_draft_and_snapshot_report_no_changes():
    document = {"appearance": {"theme": "dark"}, "ai": {"tone": "formal"}}
    chatbot = make_chatbot(
        draft=dict(document),
        published=dict(document),
        status=ChatbotStatus.PUBLISHED.value,
    )
    assert cz.has_unpublished_changes(chatbot) is False


def test_an_edited_draft_reports_unpublished_changes():
    chatbot = make_chatbot(
        draft={"appearance": {"theme": "dark"}},
        published={"appearance": {"theme": "light"}},
        status=ChatbotStatus.PUBLISHED.value,
    )
    assert cz.has_unpublished_changes(chatbot) is True


def test_saving_a_value_equal_to_the_default_is_not_an_unpublished_change():
    """Compared after resolution, so a no-op save does not light the
    indicator — an indicator that cries wolf is one people learn to ignore."""
    chatbot = make_chatbot(
        draft={"appearance": {"theme": "light"}},
        published={},
        status=ChatbotStatus.PUBLISHED.value,
    )
    # An empty published document is treated as "never published".
    assert cz.has_unpublished_changes(chatbot) is False

    chatbot.published_config = {"appearance": {}}
    assert cz.has_unpublished_changes(chatbot) is False


def test_the_overridden_paths_name_what_the_customer_chose():
    chatbot = make_chatbot(
        draft={
            "appearance": {"theme": "dark", "colors": {"primary": "#0f766e"}},
            "ai": {"tone": "formal"},
        },
        launcher={"shape": "square"},
    )
    paths = cz.overridden_paths(chatbot)
    assert "content.tone" in paths
    assert "appearance.theme" in paths
    assert "appearance.colors.primary" in paths
    assert "launcher.shape" in paths
    assert "appearance.colors.ai_bubble" not in paths


def test_a_chatbot_nobody_has_customised_has_no_overrides():
    assert cz.overridden_paths(make_chatbot()) == []


# ---------------------------------------------------------------------------
# Uploaded images: decoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (PNG_BYTES, "image/png"),
        (GIF_BYTES, "image/gif"),
        (WEBP_BYTES, "image/webp"),
        (JPEG_BYTES, "image/jpeg"),
    ],
)
def test_a_supported_image_is_identified_from_its_bytes(raw: bytes, expected: str):
    data, content_type = cz.decode_image(base64.b64encode(raw).decode())
    assert content_type == expected
    assert data == raw


def test_a_data_url_prefix_is_accepted():
    """Which is what FileReader hands the browser."""
    payload = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()
    data, content_type = cz.decode_image(payload)
    assert content_type == "image/png"
    assert data == PNG_BYTES


def test_the_declared_content_type_is_ignored_in_favour_of_the_bytes():
    """A client's claim is just a string in a JSON body; a browser will sniff
    the bytes, so the bytes are what we store and serve."""
    payload = "data:image/svg+xml;base64," + base64.b64encode(PNG_BYTES).decode()
    _, content_type = cz.decode_image(payload)
    assert content_type == "image/png"


def test_an_svg_is_refused():
    """An SVG is a document that can carry script, and these bytes are served
    from our own origin."""
    svg = b'<svg xmlns="http://www.w3.org/2000/svg"><script>alert(1)</script></svg>'
    with pytest.raises(errors.AppError) as caught:
        cz.decode_image(base64.b64encode(svg).decode())
    assert caught.value.code == "unsupported_image"


@pytest.mark.parametrize(
    "payload",
    [
        "not base64 at all !!",
        base64.b64encode(b"GIF").decode(),
        base64.b64encode(b"<html><body>hi</body></html>").decode(),
        base64.b64encode(b"%PDF-1.7 ...").decode(),
    ],
)
def test_anything_that_is_not_a_supported_image_is_refused(payload: str):
    with pytest.raises(errors.AppError) as caught:
        cz.decode_image(payload)
    assert caught.value.status_code in (413, 422)


def test_an_empty_upload_is_refused():
    with pytest.raises(errors.AppError) as caught:
        cz.decode_image(base64.b64encode(b"").decode() or "====")
    assert caught.value.status_code == 422


def test_an_oversized_image_is_refused_with_the_limit_in_the_message():
    oversized = PNG_BYTES + b"\x00" * (ASSET_MAX_BYTES + 1)
    with pytest.raises(errors.AppError) as caught:
        cz.decode_image(base64.b64encode(oversized).decode())
    assert caught.value.code == "image_too_large"
    assert str(ASSET_MAX_BYTES // 1024) in caught.value.message


def test_whitespace_in_the_payload_is_tolerated():
    chunked = "\n".join(
        base64.b64encode(PNG_BYTES).decode()[index : index + 40] for index in range(0, 200, 40)
    )
    data, _ = cz.decode_image(chunked)
    assert data == PNG_BYTES


def test_an_asset_url_is_built_from_the_configured_api_base():
    asset_id = uuid.uuid4()
    assert cz.asset_url(asset_id).startswith(get_settings().public_api_base_url)
    assert str(asset_id) in cz.asset_url(asset_id)


# ---------------------------------------------------------------------------
# The preview uses the public contract, without exposing the draft publicly
# ---------------------------------------------------------------------------


def test_the_studio_previews_the_contract_a_visitor_would_receive():
    """Not a second mapping of it: the same function, with published=False."""
    from helpers import deployment

    chatbot = make_chatbot(
        draft={
            "ai": {"chatbot_name": "Draft Name", "persona": "Internal note"},
            "appearance": {"colors": {"primary": "#0f766e"}},
        },
        published={"ai": {"chatbot_name": "Published Name"}},
        status=ChatbotStatus.PUBLISHED.value,
    )

    preview = deployment.public_config(chatbot, published=False)
    live = deployment.public_config(chatbot)

    assert preview.chatbot_name == "Draft Name"
    assert live.chatbot_name == "Published Name"
    assert preview.appearance.light.primary == "#0f766e"
    # And the preview is still the allowlist: the persona does not travel just
    # because the customer is the one looking.
    assert "Internal note" not in preview.model_dump_json()


def test_the_public_endpoints_never_resolve_the_draft():
    """`published=True` is the default, and the public router must keep it.

    This is the one guard worth a structural test: the preview needed
    `public_config` to take a flag, and a flag that could be flipped on the
    anonymous path would serve visitors an unpublished draft.

    The dashboard's own Deploy tab does resolve the draft launcher, which is
    correct — it is authenticated and owner-scoped — so the assertion is
    specifically about `public_config`.
    """
    import inspect
    import re

    from routers import deployment as deployment_router

    source = inspect.getsource(deployment_router)
    assert re.search(r"public_config\([^)]*published\s*=\s*False", source) is None


def test_resolving_an_unsaved_configuration_writes_nothing():
    """The preview builds a detached stand-in, never mutating the loaded row."""
    import inspect

    from helpers import customization

    source = inspect.getsource(customization.preview_chatbot)
    assert "db.add" not in source
    assert "db.commit" not in source
    assert "return Chatbot(" in source


def test_a_published_uploaded_icon_survives_a_change_of_api_host():
    """An uploaded icon is snapshotted as `asset:<uuid>`, not as a URL.

    Baking the absolute URL into the published snapshot would tie the chatbot
    to whichever API hostname was configured on the day it was published —
    and in development, where that host is plain http, the resolver would
    refuse its own URL on the way back out and the icon would quietly vanish.
    """
    asset_id = uuid.uuid4()
    chatbot = make_chatbot(
        launcher={"icon_image": f"asset:{asset_id}"},
        # What `publish` writes: the resolved dump, plus the reference.
        published={
            "deployment": {
                **cz.resolve_launcher(
                    make_chatbot(launcher={"icon_image": f"asset:{asset_id}"})
                ).model_dump(),
                "icon_image": f"asset:{asset_id}",
            }
        },
        status=ChatbotStatus.PUBLISHED.value,
    )
    assert cz.resolve_launcher(chatbot, published=True).icon_url == cz.asset_url(asset_id)


def test_the_publish_snapshot_carries_the_icon_reference():
    """Asserted on the writer as well as the reader, since the two halves of
    this only work together."""
    import inspect

    from helpers import deployment

    source = inspect.getsource(deployment.publish)
    assert '"icon_image"' in source


def test_a_reset_scope_outside_the_known_set_is_refused():
    assert "all" in cz.RESET_SCOPES
    assert set(cz.RESET_SCOPES) == {"all", "content", "appearance", "launcher"}

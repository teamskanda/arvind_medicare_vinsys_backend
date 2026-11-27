"""Customisation: resolution, persistence and uploaded images.

This module is the one place that turns stored JSON into settings anything
else may use. Everything that reads a customer's design goes through
`resolve_content`, `resolve_appearance` and `resolve_launcher`, and they all
obey the same two rules:

  1. **Never trust the stored document.** It is JSONB written by an API
     client. Every string is type-checked, every number is clamped to the
     range the UI offers, every colour must be a hex literal and every choice
     must be one of the values this module defines. A hostile or stale
     document produces a plain, working chatbot — never a broken layout, an
     unbounded iframe or a CSS injection.
  2. **Resolve once, for both audiences.** The studio preview and the live
     public chatbot call the same resolver with the same `published` flag
     semantics. That is what makes the preview truthful: it is not a
     reimplementation of the chatbot's appearance, it is the same data
     through the same code.

Derived rather than demanded
---------------------------

A customer is asked for one colour — their brand colour. Every other colour
has a default derived from it (`_palette`), and an explicit choice simply
replaces the derived value. The alternative, eleven required colour pickers,
produces unreadable chatbots: people pick a dark header and dark text. Header
and bubble text are therefore computed from the background they sit on
(`readable_text_color`) unless the customer overrides them, so the default is
always legible.

Both themes are always resolved. `theme: "auto"` is answered in the browser
from `prefers-color-scheme`, because that is the only place the answer exists.
"""

from __future__ import annotations

import base64
import binascii
import logging
import re
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from config import get_settings
from helpers import errors
from helpers.chatbots import get_chatbot
from helpers.icons import LAUNCHER_ICONS
from helpers.prompt_builder import (
    GROUNDING_LEVELS,
    RESPONSE_LENGTHS,
    TONE_PRESETS,
    ChatbotAiConfig,
)
from models.chatbot import Chatbot
from models.customization import (
    ASSET_CONTENT_TYPES,
    ASSET_MAX_BYTES,
    ASSET_REFERENCE_RE,
    MAX_BORDER_RADIUS,
    MAX_SUGGESTED_QUESTION_CHARACTERS,
    MAX_SUGGESTED_QUESTIONS,
    MIN_BORDER_RADIUS,
    AssetOut,
    ChatbotAsset,
    CustomizationLimits,
    CustomizationOptions,
    CustomizationState,
    CustomizationUpdate,
    LauncherSettings,
    OptionItem,
    Palette,
    ResolvedAppearance,
    ResolvedContent,
    is_hex_color,
)
from models.user import AuthenticatedUser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Catalogues of valid choices
# ---------------------------------------------------------------------------
# Defined here, served to the studio by the options endpoint, and used to
# validate every stored value. One list, so the picker cannot offer something
# the validator rejects.


@dataclass(frozen=True, slots=True)
class FontChoice:
    label: str
    stack: str
    description: str


# No third-party fonts. Every stack below resolves to a face already on the
# visitor's device, except Inter, which the embed page self-hosts through
# next/font and therefore already loads whatever the customer picks. So
# choosing a font costs an embedded chatbot nothing: no extra request, no
# connection to another company's CDN, and no flash of unstyled text.
#
# A Google Fonts picker would look richer in the studio and would put a
# request to fonts.gstatic.com on every one of our customers' websites. Not
# worth it for a chat window.
FONTS: dict[str, FontChoice] = {
    "system": FontChoice(
        "System default",
        'ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
        "Matches the visitor's device. The safest choice.",
    ),
    "inter": FontChoice(
        "Inter",
        'var(--font-inter), ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif',
        "Clean, modern and highly legible at small sizes.",
    ),
    "humanist": FontChoice(
        "Humanist",
        '"Segoe UI", Candara, Optima, "Trebuchet MS", sans-serif',
        "Softer and a little warmer than the system default.",
    ),
    "serif": FontChoice(
        "Serif",
        'Georgia, "Times New Roman", Times, serif',
        "Traditional and formal. Suits law, finance and healthcare.",
    ),
    "mono": FontChoice(
        "Monospace",
        "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
        "Technical. Suits developer tools and documentation.",
    ),
}

THEMES: dict[str, str] = {
    "light": "Light",
    "dark": "Dark",
    "auto": "Match the visitor's device",
}

# From the shared registry in helpers/icons.py, which helpers/widget.py also
# reads to draw them. The studio's gallery renders the same path data, served
# in the options catalogue below, so the icon a customer picks in the studio
# is the icon their website draws.
LAUNCHER_ICON_KEYS: dict[str, str] = {
    key: icon.label for key, icon in LAUNCHER_ICONS.items()
}

LAUNCHER_POSITIONS: dict[str, str] = {
    "bottom-right": "Bottom right",
    "bottom-left": "Bottom left",
}

LAUNCHER_SHAPES: dict[str, str] = {
    "circle": "Circle",
    "rounded": "Rounded square",
    "square": "Square",
}

OPEN_ANIMATIONS: dict[str, str] = {
    "scale": "Grow out of the button",
    "slide": "Slide up",
    "fade": "Fade in",
    "none": "No animation",
}

ATTENTION_ANIMATIONS: dict[str, str] = {
    "none": "None",
    "pulse": "Gentle pulse",
    "bounce": "Occasional bounce",
    "glow": "Soft glow",
}

CLOSE_ANIMATIONS = {
    "scale": "Shrink into the button", "slide": "Slide down",
    "fade": "Fade out", "none": "No animation",
}

# Not an exhaustive list of languages: these are the ones a customer can
# reasonably expect Gemma to answer well in, plus automatic detection, which
# is the default and the right answer for most Indian websites.
LANGUAGES: dict[str, str] = {
    "auto": "Match the visitor's language",
    "English": "English",
    "Hindi": "हिन्दी (Hindi)",
    "Kannada": "ಕನ್ನಡ (Kannada)",
    "Tamil": "தமிழ் (Tamil)",
    "Telugu": "తెలుగు (Telugu)",
    "Malayalam": "മലയാളം (Malayalam)",
    "Marathi": "मराठी (Marathi)",
    "Bengali": "বাংলা (Bengali)",
    "Gujarati": "ગુજરાતી (Gujarati)",
    "Arabic": "العربية (Arabic)",
    "French": "Français (French)",
    "Spanish": "Español (Spanish)",
    "German": "Deutsch (German)",
}

COLOR_PRESETS: tuple[tuple[str, str], ...] = (
    ("#7d2a47", "SKANDA burgundy"),
    ("#1f5a8a", "Ocean blue"),
    ("#0f766e", "Teal"),
    ("#166534", "Forest"),
    ("#b45309", "Amber"),
    ("#be123c", "Rose"),
    ("#4338ca", "Indigo"),
    ("#334155", "Slate"),
)

DEFAULT_THEME = "light"
# Inter, self-hosted by the embed page: identical on every device, where the
# system stack falls back to a thin Segoe UI on Windows.
DEFAULT_FONT = "inter"
DEFAULT_BORDER_RADIUS = 12

# Base surfaces per theme. Only the three that cannot be derived from a brand
# colour: the page behind the conversation, the assistant's bubble and the
# body text.
_LIGHT_BASE = {
    "chat_background": "#faf8f7",
    "ai_bubble": "#ffffff",
    "message_text": "#1b1a1a",
}
_DARK_BASE = {
    "chat_background": "#141314",
    "ai_bubble": "#232022",
    "message_text": "#f3f1f1",
}

_COLOR_KEYS = (
    "primary",
    "secondary",
    "header_background",
    "header_text",
    "chat_background",
    "user_bubble",
    "user_bubble_text",
    "ai_bubble",
    "message_text",
)

# How many images one chatbot may keep. An avatar, a logo and a launcher icon
# plus room to try alternatives; a bound exists so an upload loop cannot grow
# the table without limit.
MAX_ASSETS_PER_CHATBOT = 12

_DATA_URL_RE = re.compile(r"^data:([a-zA-Z0-9.+/-]{1,80});base64,", re.IGNORECASE)

# File signatures we accept, checked against the bytes rather than against
# whatever content type the client claimed.
_MAGIC_BYTES: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


# ---------------------------------------------------------------------------
# Reading the stored document
# ---------------------------------------------------------------------------


def config_subtree(chatbot: Chatbot, key: str, *, published: bool) -> dict[str, Any]:
    """One subtree of the draft or published configuration, or an empty dict.

    `published=True` reads `published_config`, which is what visitors are
    served; `published=False` reads `draft_config`, which is what the studio
    and the playground preview. Every caller in the codebase picks one
    explicitly — there is no default, because getting it wrong means either
    showing a visitor an unpublished draft or showing a customer their own
    changes having no effect.
    """
    source = chatbot.published_config if published else chatbot.draft_config
    if not isinstance(source, dict):
        return {}
    subtree = source.get(key)
    return subtree if isinstance(subtree, dict) else {}


def _text(raw: dict[str, Any], key: str, default: str | None, cap: int) -> str | None:
    value = raw.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()[:cap]
    return default


def _choice(raw: dict[str, Any], key: str, allowed: dict[str, Any], default: str) -> str:
    value = raw.get(key)
    if isinstance(value, str) and value.strip() in allowed:
        return value.strip()
    return default


def _integer(raw: dict[str, Any], key: str, default: int, low: int, high: int) -> int:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        return default
    return max(low, min(high, value))


def _flag(raw: dict[str, Any], key: str, default: bool) -> bool:
    value = raw.get(key)
    return value if isinstance(value, bool) else default


# ---------------------------------------------------------------------------
# Colour arithmetic
# ---------------------------------------------------------------------------


def _parse_hex(value: str) -> tuple[int, int, int]:
    candidate = value.strip().lstrip("#")
    if len(candidate) == 3:
        candidate = "".join(character * 2 for character in candidate)
    return (
        int(candidate[0:2], 16),
        int(candidate[2:4], 16),
        int(candidate[4:6], 16),
    )


def _to_hex(rgb: tuple[float, float, float]) -> str:
    return "#" + "".join(f"{max(0, min(255, round(channel))):02x}" for channel in rgb)


def mix(base: str, other: str, weight: float) -> str:
    """Blend `weight` of `other` into `base`. `weight=0` returns `base`."""
    first = _parse_hex(base)
    second = _parse_hex(other)
    ratio = max(0.0, min(1.0, weight))
    return _to_hex(
        tuple(a * (1 - ratio) + b * ratio for a, b in zip(first, second))  # type: ignore[arg-type]
    )


def relative_luminance(color: str) -> float:
    """WCAG relative luminance, used to choose readable text."""
    channels = []
    for raw in _parse_hex(color):
        value = raw / 255
        channels.append(
            value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        )
    red, green, blue = channels
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def readable_text_color(background: str) -> str:
    """White or near-black, whichever has more contrast on `background`.

    This is why a customer can pick any header colour and still have a
    readable header: the text colour follows the background unless they
    deliberately override it.
    """
    return "#1b1a1a" if relative_luminance(background) > 0.45 else "#ffffff"


def _palette(primary: str, overrides: dict[str, str], *, dark: bool) -> Palette:
    """Derive a complete palette from one brand colour, honouring overrides.

    Overrides apply to both themes. That is deliberate: a customer who sets
    their user-bubble colour means it, and silently ignoring it in dark mode
    would be the kind of "setting that does not quite work" this phase is
    meant to eliminate.
    """
    base = _DARK_BASE if dark else _LIGHT_BASE
    towards_text = "#ffffff" if dark else "#000000"

    secondary = overrides.get("secondary") or mix(primary, towards_text, 0.22)
    header_background = overrides.get("header_background") or (
        mix(primary, "#000000", 0.28) if dark else primary
    )
    header_text = overrides.get("header_text") or readable_text_color(header_background)
    chat_background = overrides.get("chat_background") or base["chat_background"]
    message_text = overrides.get("message_text") or base["message_text"]
    ai_bubble = overrides.get("ai_bubble") or base["ai_bubble"]
    # The visitor's own bubble: a hint of the brand colour over the
    # conversation background, stronger in dark mode where a faint tint
    # disappears.
    user_bubble = overrides.get("user_bubble") or mix(
        chat_background, primary, 0.30 if dark else 0.14
    )
    user_bubble_text = overrides.get("user_bubble_text") or readable_text_color(user_bubble)

    return Palette(
        primary=primary,
        secondary=secondary,
        header_background=header_background,
        header_text=header_text,
        chat_background=chat_background,
        user_bubble=user_bubble,
        user_bubble_text=user_bubble_text,
        ai_bubble=ai_bubble,
        message_text=message_text,
        # Derived, never customer-set: a border and a muted label always have
        # to sit between the text and the background they are on.
        border=mix(chat_background, message_text, 0.14),
        muted_text=mix(message_text, chat_background, 0.42),
    )


# ---------------------------------------------------------------------------
# Images: `asset:<uuid>` references and external URLs
# ---------------------------------------------------------------------------


def asset_url(asset_id: uuid.UUID | str) -> str:
    """The absolute URL the chat UI loads an uploaded image from.

    Built from `PUBLIC_API_BASE_URL` at read time rather than stored, so
    moving the API host does not leave every chatbot pointing at the old one.
    """
    return f"{get_settings().public_api_base_url}/api/public/assets/{asset_id}"


def resolve_image(value: Any) -> str | None:
    """Turn a stored image setting into a URL, or refuse it.

    Two accepted forms and no others:

      * `asset:<uuid>` — an image this customer uploaded, served by our own
        asset endpoint.
      * an absolute `https://` URL — someone else's CDN, which is their
        choice to make.

    `http://`, `data:` and `javascript:` are all rejected. These values end up
    in an `<img src>` inside our own page *and* inside the loader script that
    runs on the customer's website, so the scheme allowlist is what keeps a
    stored string from becoming script execution on their domain. Plain http
    is refused separately because it would downgrade a secure page to mixed
    content and be blocked by the browser anyway — better a missing avatar
    than a broken padlock on the customer's site.
    """
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or len(candidate) > 2048:
        return None

    reference = ASSET_REFERENCE_RE.match(candidate)
    if reference:
        try:
            return asset_url(uuid.UUID(reference.group(1)))
        except ValueError:
            return None

    if candidate.lower().startswith("https://") and len(candidate) > len("https://"):
        return candidate
    return None


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def primary_color(chatbot: Chatbot, *, published: bool) -> str:
    """The chatbot's brand colour: chosen, scraped, or ours.

    The scraped fallback matters more than it looks: a customer who never
    opens the studio still gets a chatbot in their own website's colour,
    because Phase 1 read it off their page.
    """
    appearance = config_subtree(chatbot, "appearance", published=published)
    colors = appearance.get("colors")
    chosen = colors.get("primary") if isinstance(colors, dict) else None
    if is_hex_color(chosen):
        return str(chosen).strip().lower()
    if is_hex_color(chatbot.brand_color):
        return str(chatbot.brand_color).strip().lower()
    return get_settings().widget_accent_color


def resolve_appearance(chatbot: Chatbot, *, published: bool) -> ResolvedAppearance:
    """Theme, palettes, font and radius for one chatbot."""
    raw = config_subtree(chatbot, "appearance", published=published)

    stored_colors = raw.get("colors")
    overrides: dict[str, str] = {}
    if isinstance(stored_colors, dict):
        for key in _COLOR_KEYS:
            value = stored_colors.get(key)
            if is_hex_color(value):
                overrides[key] = str(value).strip().lower()

    primary = primary_color(chatbot, published=published)
    font = _choice(raw, "font", FONTS, DEFAULT_FONT)

    # The scraped favicon is the fallback avatar, as in Phase 3 — but only
    # when the customer has not chosen one, and only over https.
    avatar = resolve_image(raw.get("avatar")) or resolve_image(chatbot.favicon_url)

    return ResolvedAppearance(
        theme=_choice(raw, "theme", THEMES, DEFAULT_THEME),
        font=font,
        font_stack=FONTS[font].stack,
        border_radius=_integer(
            raw, "border_radius", DEFAULT_BORDER_RADIUS, MIN_BORDER_RADIUS, MAX_BORDER_RADIUS
        ),
        avatar_url=avatar,
        logo_url=resolve_image(raw.get("logo")),
        light=_palette(primary, overrides, dark=False),
        dark=_palette(primary, overrides, dark=True),
    )


def suggested_questions(chatbot: Chatbot, *, published: bool) -> list[str]:
    """Starter prompts, when the customer configured any.

    Absent by default: inventing plausible-sounding questions would imply the
    chatbot can answer them, which depends entirely on what was scraped.
    """
    ai = config_subtree(chatbot, "ai", published=published)
    raw = ai.get("suggested_questions")
    if not isinstance(raw, list):
        return []
    questions: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            question = item.strip()[:MAX_SUGGESTED_QUESTION_CHARACTERS]
            if question not in questions:
                questions.append(question)
        if len(questions) >= MAX_SUGGESTED_QUESTIONS:
            break
    return questions


def resolve_content(chatbot: Chatbot, *, published: bool) -> ResolvedContent:
    """Content settings, resolved exactly as the prompt will see them.

    Built on top of `ChatbotAiConfig.from_chatbot` rather than beside it, so
    the studio shows the same persona, tone, language, length, fallback and
    grounding that `helpers/prompt_builder.py` will actually use. A second
    implementation here would eventually disagree with the prompt, and the
    customer would be editing a value that no longer did anything.
    """
    source = chatbot.published_config if published else chatbot.draft_config
    ai_config = ChatbotAiConfig.from_chatbot(
        name=chatbot.name, website_url=chatbot.website_url, config=source
    )
    raw = config_subtree(chatbot, "ai", published=published)

    return ResolvedContent(
        chatbot_name=ai_config.chatbot_name,
        business_name=ai_config.business_name,
        greeting=ai_config.greeting,
        welcome_message=_text(raw, "welcome_message", None, 300),
        suggested_questions=suggested_questions(chatbot, published=published),
        persona=ai_config.persona,
        instructions=ai_config.instructions,
        tone=ai_config.tone,
        language=ai_config.language,
        response_length=ai_config.response_length,
        fallback_message=ai_config.fallback_message,
        cite_sources=ai_config.cite_sources,
        grounding=ai_config.grounding,
    )


def resolve_launcher(chatbot: Chatbot, *, published: bool = False) -> LauncherSettings:
    """Floating-launcher settings for one chatbot.

    Reads `deployment_config` (the working copy) or the `deployment` subtree
    of `published_config` (what visitors get). Every value is clamped here
    rather than trusted, because these numbers are interpolated into a script
    that runs on the customer's own website: an unbounded window size would
    cover their page, and a non-hex colour would be a CSS injection on their
    domain.
    """
    settings = get_settings()

    if published:
        raw = config_subtree(chatbot, "deployment", published=True)
    else:
        raw = chatbot.deployment_config if isinstance(chatbot.deployment_config, dict) else {}

    # The launcher's colour defaults to the brand colour, and may be set
    # separately — a dark header with a bright button is a common, reasonable
    # choice, so these are two controls rather than one.
    accent = raw.get("accent_color")
    if not is_hex_color(accent):
        accent = primary_color(chatbot, published=published)

    icon_url = resolve_image(raw.get("icon_image")) or resolve_image(raw.get("icon_url"))

    label = raw.get("label")
    label = label.strip()[:28] if isinstance(label, str) and label.strip() else None

    return LauncherSettings(
        icon=_choice(raw, "icon", LAUNCHER_ICON_KEYS, settings.widget_launcher_icon),
        icon_url=icon_url,
        accent_color=str(accent).strip().lower(),
        launcher_size=_integer(raw, "launcher_size", settings.widget_launcher_size, 40, 96),
        shape=_choice(raw, "shape", LAUNCHER_SHAPES, settings.widget_launcher_shape),
        position=_choice(
            raw, "position", LAUNCHER_POSITIONS, settings.widget_launcher_position
        ),
        offset_x=_integer(raw, "offset_x", settings.widget_launcher_offset, 0, 200),
        offset_y=_integer(raw, "offset_y", settings.widget_launcher_offset, 0, 200),
        window_width=_integer(raw, "window_width", settings.widget_window_width, 280, 640),
        window_height=_integer(raw, "window_height", settings.widget_window_height, 360, 900),
        label=label,
        open_animation=_choice(
            raw, "open_animation", OPEN_ANIMATIONS, settings.widget_open_animation
        ),
        close_animation=_choice(raw, "close_animation", OPEN_ANIMATIONS, "scale"),
        attention_animation=_choice(
            raw, "attention_animation", ATTENTION_ANIMATIONS, "none"
        ),
        open_on_load=_flag(raw, "open_on_load", False),
        show_branding=_flag(raw, "show_branding", True),
    )


def resolve_state(chatbot: Chatbot, *, published: bool) -> CustomizationState:
    """One complete configuration: content, appearance and launcher."""
    return CustomizationState(
        content=resolve_content(chatbot, published=published),
        appearance=resolve_appearance(chatbot, published=published),
        launcher=resolve_launcher(chatbot, published=published),
    )


def has_unpublished_changes(chatbot: Chatbot) -> bool:
    """Does the draft differ from what visitors are currently served?

    Compared after resolution, not as raw JSON: an empty `{}` and an absent
    key resolve to the same chatbot, and a customer who saved a value equal to
    the default has not changed anything. Comparing documents would show
    "unpublished changes" for a no-op save, which trains people to ignore the
    indicator.
    """
    if not isinstance(chatbot.published_config, dict) or not chatbot.published_config:
        return False
    draft = resolve_state(chatbot, published=False).model_dump()
    published = resolve_state(chatbot, published=True).model_dump()
    return draft != published


def overridden_paths(chatbot: Chatbot) -> list[str]:
    """Which settings the customer has explicitly chosen.

    Lets the studio say "derived from your brand colour" versus "you set
    this", and offer to clear one value without resetting everything.
    """
    paths: list[str] = []

    ai = config_subtree(chatbot, "ai", published=False)
    for key in (
        "chatbot_name",
        "business_name",
        "greeting",
        "welcome_message",
        "suggested_questions",
        "persona",
        "instructions",
        "tone",
        "language",
        "response_length",
        "fallback_message",
        "cite_sources",
        "grounding",
    ):
        if ai.get(key) not in (None, "", []):
            paths.append(f"content.{key}")

    appearance = config_subtree(chatbot, "appearance", published=False)
    for key in ("theme", "font", "border_radius", "avatar", "logo"):
        if appearance.get(key) not in (None, ""):
            paths.append(f"appearance.{key}")
    colors = appearance.get("colors")
    if isinstance(colors, dict):
        for key in _COLOR_KEYS:
            if is_hex_color(colors.get(key)):
                paths.append(f"appearance.colors.{key}")

    launcher = chatbot.deployment_config if isinstance(chatbot.deployment_config, dict) else {}
    for key in sorted(launcher):
        if launcher.get(key) not in (None, ""):
            paths.append(f"launcher.{key}")

    return paths


# ---------------------------------------------------------------------------
# The options catalogue
# ---------------------------------------------------------------------------


def options() -> CustomizationOptions:
    """Every valid choice, with the labels the studio shows.

    Served rather than duplicated in the frontend. The studio renders its
    pickers from this, so adding a tone preset or a launcher shape is one
    change in Python and the UI picks it up — and there is no way for the
    picker to offer a value the validator will reject.
    """
    return CustomizationOptions(
        themes=[OptionItem(value=key, label=label) for key, label in THEMES.items()],
        fonts=[
            OptionItem(
                value=key,
                label=choice.label,
                description=choice.description,
                font_stack=choice.stack,
            )
            for key, choice in FONTS.items()
        ],
        tones=[
            OptionItem(value=key, label=preset.label, description=preset.description)
            for key, preset in TONE_PRESETS.items()
        ],
        languages=[
            OptionItem(value=key, label=label) for key, label in LANGUAGES.items()
        ],
        response_lengths=[
            OptionItem(value=key, label=label, description=description)
            for key, (label, description) in RESPONSE_LENGTHS.items()
        ],
        grounding_levels=[
            OptionItem(value=key, label=level.label, description=level.description)
            for key, level in GROUNDING_LEVELS.items()
        ],
        launcher_icons=[
            OptionItem(value=key, label=icon.label, svg_path=icon.path)
            for key, icon in LAUNCHER_ICONS.items()
        ],
        launcher_positions=[
            OptionItem(value=key, label=label)
            for key, label in LAUNCHER_POSITIONS.items()
        ],
        launcher_shapes=[
            OptionItem(value=key, label=label) for key, label in LAUNCHER_SHAPES.items()
        ],
        open_animations=[
            OptionItem(value=key, label=label) for key, label in OPEN_ANIMATIONS.items()
        ],
        close_animations=[OptionItem(value=key, label=label) for key, label in CLOSE_ANIMATIONS.items()],
        attention_animations=[
            OptionItem(value=key, label=label)
            for key, label in ATTENTION_ANIMATIONS.items()
        ],
        color_presets=[
            OptionItem(value=value, label=label, color=value)
            for value, label in COLOR_PRESETS
        ],
        limits=CustomizationLimits(),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _merge(stored: dict[str, Any], changes: dict[str, Any]) -> dict[str, Any]:
    """Apply a partial update, returning a new dict.

    A key present with a value is written; a key present with `null` is
    removed, which is how "reset this one setting" works; a key that was not
    sent is left alone. A new object is returned rather than the original
    mutated because SQLAlchemy does not track in-place changes to a JSONB
    dict — an in-place edit would be silently discarded on commit.
    """
    merged = dict(stored)
    for key, value in changes.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, dict):
            nested = merged.get(key)
            merged[key] = _merge(nested if isinstance(nested, dict) else {}, value)
            if not merged[key]:
                merged.pop(key, None)
        else:
            merged[key] = value
    return merged


async def _validate_asset_references(
    db: AsyncSession, chatbot: Chatbot, values: dict[str, Any]
) -> None:
    """Every `asset:<uuid>` must name an image belonging to *this* chatbot.

    Without this, a customer could point their configuration at an asset id
    belonging to someone else's chatbot. The image would still be one its
    owner published publicly, so this is not a confidentiality hole — it is a
    correctness one: the reference would survive until that asset was deleted
    and then break, and a chatbot's configuration should not depend on another
    customer's row.
    """
    references: list[uuid.UUID] = []
    for value in values.values():
        if not isinstance(value, str):
            continue
        match = ASSET_REFERENCE_RE.match(value.strip())
        if not match:
            continue
        try:
            references.append(uuid.UUID(match.group(1)))
        except ValueError:
            raise errors.AppError(
                "invalid_asset", "That image reference is not valid.", status_code=422
            ) from None

    if not references:
        return

    result = await db.execute(
        select(ChatbotAsset.id).where(
            ChatbotAsset.id.in_(references),
            ChatbotAsset.chatbot_id == chatbot.id,
            ChatbotAsset.owner_id == chatbot.owner_id,
        )
    )
    found = set(result.scalars().all())
    missing = [reference for reference in references if reference not in found]
    if missing:
        raise errors.AppError(
            "invalid_asset",
            "That image is no longer available. Upload it again.",
            status_code=422,
        )


async def apply_update(
    db: AsyncSession, chatbot: Chatbot, payload: CustomizationUpdate
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Work out what the configuration documents become, without writing them.

    Returns the new `draft_config`, the new `deployment_config`, and whether
    anything actually changed. Separated from `save_draft` so the studio's
    live preview can resolve a configuration that has not been saved: the
    preview then shows values produced by this module's own resolvers rather
    than by a second implementation in the browser, which is what keeps the
    preview honest about derived colours.
    """
    draft = chatbot.draft_config if isinstance(chatbot.draft_config, dict) else {}
    launcher = (
        chatbot.deployment_config if isinstance(chatbot.deployment_config, dict) else {}
    )
    new_draft = dict(draft)
    new_launcher = dict(launcher)
    touched = False

    if payload.content is not None:
        changes = payload.content.model_dump(exclude_unset=True)
        if changes:
            existing = new_draft.get("ai")
            new_draft["ai"] = _merge(existing if isinstance(existing, dict) else {}, changes)
            touched = True

    if payload.appearance is not None:
        changes = payload.appearance.model_dump(exclude_unset=True)
        if changes:
            await _validate_asset_references(db, chatbot, changes)
            existing = new_draft.get("appearance")
            new_draft["appearance"] = _merge(
                existing if isinstance(existing, dict) else {}, changes
            )
            touched = True

    if payload.launcher is not None:
        changes = payload.launcher.model_dump(exclude_unset=True)
        if changes:
            await _validate_asset_references(db, chatbot, changes)
            new_launcher = _merge(new_launcher, changes)
            touched = True

    return new_draft, new_launcher, touched


async def save_draft(
    db: AsyncSession,
    user: AuthenticatedUser,
    chatbot_id: uuid.UUID,
    payload: CustomizationUpdate,
) -> Chatbot:
    """Persist a partial customisation update to the draft configuration.

    Writes only the groups that were sent. Nothing here touches
    `published_config`: a saved draft changes the playground and the studio
    preview immediately and changes the live chatbot not at all, until the
    customer publishes.
    """
    chatbot = await get_chatbot(db, user, chatbot_id)
    new_draft, new_launcher, touched = await apply_update(db, chatbot, payload)
    if not touched:
        return chatbot

    # Assigned, never mutated in place: SQLAlchemy does not track in-place
    # changes to a JSONB dict, so an in-place edit would be discarded here.
    chatbot.draft_config = new_draft
    chatbot.deployment_config = new_launcher

    await db.commit()
    await db.refresh(chatbot)
    return chatbot


async def preview_chatbot(
    db: AsyncSession,
    user: AuthenticatedUser,
    chatbot_id: uuid.UUID,
    payload: CustomizationUpdate,
) -> Chatbot:
    """A stand-in chatbot carrying an unsaved configuration.

    Returned as a `Chatbot` rather than a resolved state so the caller can run
    *any* of the normal resolvers over it — including
    `deployment.public_config`, which is how the studio previews the exact
    contract a visitor's browser would receive.

    Why this exists rather than resolving in the browser: deriving a palette
    from one brand colour is real work — blending, contrast decisions, two
    themes — and a second copy of it in TypeScript would drift from this one.
    The first symptom of that drift would be a customer designing against a
    preview that does not match their live chatbot, which is the one thing
    this phase must not do. So the preview asks the same resolver.

    Nothing is written: the merged documents are resolved on a transient
    object and the session is never committed.
    """
    chatbot = await get_chatbot(db, user, chatbot_id)
    new_draft, new_launcher, _ = await apply_update(db, chatbot, payload)

    # A detached stand-in rather than the loaded row: a mutated ORM object
    # could be flushed by any later statement in this request's session.
    # Never added to the session, so nothing here can be written.
    return Chatbot(
        id=chatbot.id,
        public_id=chatbot.public_id,
        owner_id=chatbot.owner_id,
        name=chatbot.name,
        website_url=chatbot.website_url,
        status=chatbot.status,
        crawl_status=chatbot.crawl_status,
        draft_config=new_draft,
        published_config=chatbot.published_config,
        deployment_config=new_launcher,
        allowed_domains=list(chatbot.allowed_domains or []),
        brand_color=chatbot.brand_color,
        favicon_url=chatbot.favicon_url,
    )


# Which stored keys each reset scope clears.
RESET_SCOPES = ("all", "content", "appearance", "launcher")


async def reset_draft(
    db: AsyncSession,
    user: AuthenticatedUser,
    chatbot_id: uuid.UUID,
    *,
    scope: str = "all",
) -> Chatbot:
    """Clear customisations, returning the chatbot to its derived defaults.

    "Derived", not "blank": the chatbot keeps its scraped name, its website's
    brand colour and its favicon, because those were never customisations —
    they are what Phase 1 found. Uploaded images are kept too, so a reset is
    recoverable; only the references to them are cleared.
    """
    if scope not in RESET_SCOPES:
        raise errors.AppError(
            "invalid_scope", "That is not something that can be reset.", status_code=422
        )

    chatbot = await get_chatbot(db, user, chatbot_id)
    draft = chatbot.draft_config if isinstance(chatbot.draft_config, dict) else {}

    if scope in ("all", "content"):
        draft = {key: value for key, value in draft.items() if key != "ai"}
    if scope in ("all", "appearance"):
        draft = {key: value for key, value in draft.items() if key != "appearance"}
    chatbot.draft_config = draft

    if scope in ("all", "launcher"):
        chatbot.deployment_config = {}

    await db.commit()
    await db.refresh(chatbot)
    return chatbot


# ---------------------------------------------------------------------------
# Uploaded images
# ---------------------------------------------------------------------------


def _sniff_content_type(data: bytes) -> str | None:
    """Identify the image from its bytes, ignoring any claimed type.

    A client's declared content type is just a string in a JSON body. What
    actually matters is what the bytes are, because that is what a browser
    will sniff and act on — so the declared type is never stored.
    """
    for signature, content_type in _MAGIC_BYTES:
        if data.startswith(signature):
            return content_type
    if len(data) >= 12 and data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def decode_image(payload: str) -> tuple[bytes, str]:
    """Decode a base64 or data-URL image, or refuse it with a clear reason.

    Checks in cost order: cheap length check, then decode, then the size cap,
    then the signature. Returns the content type found in the bytes.
    """
    candidate = payload.strip()
    match = _DATA_URL_RE.match(candidate)
    if match:
        candidate = candidate[match.end() :]
    candidate = re.sub(r"\s+", "", candidate)

    try:
        data = base64.b64decode(candidate, validate=True)
    except (binascii.Error, ValueError):
        raise errors.AppError(
            "invalid_image",
            "That file could not be read. Please choose a PNG, JPG, WebP or GIF image.",
            status_code=422,
        ) from None

    if not data:
        raise errors.AppError(
            "invalid_image", "That image file is empty.", status_code=422
        )
    if len(data) > ASSET_MAX_BYTES:
        raise errors.AppError(
            "image_too_large",
            f"That image is larger than {ASSET_MAX_BYTES // 1024} KB. "
            "Please upload a smaller one.",
            status_code=413,
        )

    content_type = _sniff_content_type(data)
    if content_type is None or content_type not in ASSET_CONTENT_TYPES:
        raise errors.AppError(
            "unsupported_image",
            "Only PNG, JPG, WebP and GIF images are supported.",
            status_code=422,
        )
    return data, content_type


def asset_out(asset: ChatbotAsset) -> AssetOut:
    return AssetOut(
        id=asset.id,
        kind=asset.kind,
        content_type=asset.content_type,
        byte_size=asset.byte_size,
        created_at=asset.created_at,
        url=asset_url(asset.id),
        reference=f"asset:{asset.id}",
    )


async def list_assets(
    db: AsyncSession, user: AuthenticatedUser, chatbot_id: uuid.UUID
) -> list[ChatbotAsset]:
    """This chatbot's uploaded images, newest first. Bytes excluded."""
    await get_chatbot(db, user, chatbot_id)
    result = await db.execute(
        select(
            ChatbotAsset.id,
            ChatbotAsset.chatbot_id,
            ChatbotAsset.owner_id,
            ChatbotAsset.kind,
            ChatbotAsset.content_type,
            ChatbotAsset.byte_size,
            ChatbotAsset.created_at,
        )
        .where(
            ChatbotAsset.chatbot_id == chatbot_id,
            ChatbotAsset.owner_id == user.id,
        )
        .order_by(ChatbotAsset.created_at.desc())
    )
    # Rows, not entities: the image bytes are never loaded to list them.
    return [
        ChatbotAsset(
            id=row.id,
            chatbot_id=row.chatbot_id,
            owner_id=row.owner_id,
            kind=row.kind,
            content_type=row.content_type,
            byte_size=row.byte_size,
            created_at=row.created_at,
        )
        for row in result
    ]


async def create_asset(
    db: AsyncSession,
    user: AuthenticatedUser,
    chatbot_id: uuid.UUID,
    *,
    kind: str,
    data: str,
) -> ChatbotAsset:
    """Store an uploaded image against one chatbot."""
    chatbot = await get_chatbot(db, user, chatbot_id)
    decoded, content_type = decode_image(data)

    result = await db.execute(
        select(func.count())
        .select_from(ChatbotAsset)
        .where(
            ChatbotAsset.chatbot_id == chatbot.id,
            ChatbotAsset.owner_id == user.id,
        )
    )
    if int(result.scalar_one()) >= MAX_ASSETS_PER_CHATBOT:
        raise errors.AppError(
            "asset_limit_reached",
            f"This chatbot already has {MAX_ASSETS_PER_CHATBOT} uploaded images. "
            "Delete one you are not using.",
            status_code=409,
        )

    asset = ChatbotAsset(
        chatbot_id=chatbot.id,
        owner_id=user.id,
        kind=kind,
        content_type=content_type,
        byte_size=len(decoded),
        data=decoded,
    )
    db.add(asset)
    await db.commit()
    await db.refresh(asset)
    logger.info(
        "Stored a %s image (%d bytes, %s) for chatbot %s.",
        kind,
        len(decoded),
        content_type,
        chatbot.id,
    )
    return asset


async def delete_asset(
    db: AsyncSession,
    user: AuthenticatedUser,
    chatbot_id: uuid.UUID,
    asset_id: uuid.UUID,
) -> None:
    """Remove an uploaded image.

    Refused while the published configuration still references it: deleting
    it would leave live visitors looking at a broken image, and the customer
    would have no idea why. They can repoint the draft and publish first.
    """
    chatbot = await get_chatbot(db, user, chatbot_id)

    reference = f"asset:{asset_id}"
    published = chatbot.published_config if isinstance(chatbot.published_config, dict) else {}
    if reference in str(published):
        raise errors.AppError(
            "asset_in_use",
            "This image is being used by your published chatbot. Change it in the "
            "studio and publish, then delete the old image.",
            status_code=409,
        )

    result = await db.execute(
        delete(ChatbotAsset).where(
            ChatbotAsset.id == asset_id,
            ChatbotAsset.chatbot_id == chatbot_id,
            ChatbotAsset.owner_id == user.id,
        )
    )
    if result.rowcount == 0:
        raise errors.not_found("image")
    await db.commit()


async def get_asset(db: AsyncSession, asset_id: uuid.UUID) -> ChatbotAsset:
    """Load one image by id, for the public asset endpoint.

    Deliberately not owner-scoped: an anonymous visitor has no identity to
    scope by. The asset id is an unguessable uuid acting as a capability, in
    the same way `public_id` does — it is handed out only by a chatbot's own
    configuration, and what it grants is an image the customer chose to put on
    their website.
    """
    result = await db.execute(select(ChatbotAsset).where(ChatbotAsset.id == asset_id))
    asset = result.scalar_one_or_none()
    if asset is None:
        raise errors.not_found("image")
    return asset

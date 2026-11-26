"""Customisation studio: the uploaded-asset table and the settings contract.

Where a setting lives, and why
------------------------------

Everything a customer can change is stored in columns that existed before
Phase 4, in two subtrees plus one column:

    chatbots.draft_config["ai"]         content  — what the chatbot says
    chatbots.draft_config["appearance"] looks    — how the chat window looks
    chatbots.deployment_config          launcher — the floating button

The line between "content" and "looks" is whether the value affects what the
chatbot *says* or what it *looks like*. The chatbot name, business name,
greeting, welcome message, suggested questions, persona, instructions, tone,
language, response length, fallback wording, citation visibility and
knowledge-grounding strictness are content: they all end up in the prompt or
in the words on screen. Theme, colours, font, corner radius, avatar and logo
are looks. The launcher keeps its own column because `helpers/widget.py`
reads it on its own to generate the loader script, with no interest in the
rest.

`chatbots.published_config` is the snapshot of all three, written by
`helpers/deployment.publish`. Visitors are served that snapshot and never the
draft, which is the whole point of the split: a customer can redesign all
afternoon without touching what is live.

Three input / output pairs
--------------------------

  * `CustomizationUpdate` is the request body. Every field is optional and
    only the fields actually sent are written (PATCH semantics), so the studio
    can save one slider without resending forty values. An explicit `null`
    clears a setting back to its default — which is how "Reset this colour"
    works.
  * `CustomizationState` is the resolved result: every value filled in, every
    colour a validated hex literal, every number clamped. Both the studio and
    the live chatbot render from this, so what the customer previews is
    produced by the same resolver the visitor's browser gets.
  * `CustomizationOptions` is the catalogue of choices — fonts, tones, icons,
    animations, limits. It is served rather than hardcoded in the frontend so
    there is one list of valid values, and the backend validator and the UI
    cannot drift apart.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator
from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Integer, LargeBinary, String, func, text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from models.base import ApiSchema, Base, ChatbotStatus
from models.message import ChatRequest

# ---------------------------------------------------------------------------
# Limits. Public so the options catalogue can report them to the studio and
# the tests can assert against one definition.
# ---------------------------------------------------------------------------

MAX_BUSINESS_NAME_CHARACTERS = 120
MAX_GREETING_CHARACTERS = 300
MAX_WELCOME_CHARACTERS = 300
MAX_SUGGESTED_QUESTIONS = 6
MAX_SUGGESTED_QUESTION_CHARACTERS = 120
MAX_LAUNCHER_LABEL_CHARACTERS = 28

MIN_BORDER_RADIUS = 0
MAX_BORDER_RADIUS = 24

# An uploaded image. 256 KB is far more than a 96px avatar needs and small
# enough that the bytes can live in PostgreSQL without a second thought.
ASSET_MAX_BYTES = 256 * 1024
# The base64 payload is ~4/3 of the bytes; the margin covers the data-URL
# prefix and any whitespace a client inserts.
ASSET_MAX_BASE64_CHARACTERS = (ASSET_MAX_BYTES * 4 // 3) + 2048

ASSET_KINDS = ("avatar", "logo", "launcher_icon")

# SVG is deliberately absent. An SVG is a document: served from our own
# origin it can run script if a visitor opens the asset URL directly, which
# would be a stored-XSS hole on our domain for the sake of a sharper logo.
ASSET_CONTENT_TYPES = ("image/png", "image/jpeg", "image/webp", "image/gif")

AssetKind = Literal["avatar", "logo", "launcher_icon"]

# `asset:<uuid>` — an image this customer uploaded. The alternative accepted
# form is an absolute https URL. Nothing else: see `helpers/customization`.
ASSET_REFERENCE_RE = re.compile(r"^asset:([0-9a-fA-F-]{36})$")

_HEX_COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def is_hex_color(value: Any) -> bool:
    """`#rgb` or `#rrggbb`, and nothing else.

    Every colour here is interpolated into inline styles — in our own pages
    and, for the launcher, into a script that runs on the customer's website.
    A hex literal cannot carry a `url(...)`, an `expression(...)` or a stray
    `;`, so restricting the alphabet removes the injection question entirely
    rather than answering it. Named colours and `rgb()` buy nothing.
    """
    return isinstance(value, str) and bool(_HEX_COLOR_RE.match(value.strip()))


# ---------------------------------------------------------------------------
# ORM: uploaded images
# ---------------------------------------------------------------------------


class ChatbotAsset(Base):
    """One uploaded image belonging to one chatbot.

    Rows are immutable. Replacing an avatar inserts a new row and repoints the
    draft configuration at it, which is what lets a published snapshot keep
    serving the image it was published with while the customer tries another
    one in the studio.
    """

    __tablename__ = "chatbot_assets"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('avatar', 'logo', 'launcher_icon')", name="chatbot_assets_kind_check"
        ),
        CheckConstraint("byte_size > 0", name="chatbot_assets_byte_size_check"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    chatbot_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chatbots.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Denormalised, as on conversations and messages: ownership stays a single
    # predicate instead of a join.
    owner_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("profiles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    kind: Mapped[str] = mapped_column(String, nullable=False)
    content_type: Mapped[str] = mapped_column(String, nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ChatbotAsset id={self.id} kind={self.kind!r}>"


# ---------------------------------------------------------------------------
# Request: partial update
# ---------------------------------------------------------------------------


def _clean_text(value: str | None) -> str | None:
    """Trim, and treat an empty result as "unset".

    An empty string and an explicit null both mean "clear this and use the
    default", so normalising here leaves the resolver one case instead of two.
    Length is already enforced by each field's `max_length`: too long is a
    refusal with a message the customer can act on, not a silent truncation.
    """
    if value is None:
        return None
    trimmed = value.strip()
    return trimmed or None


class ContentSettingsInput(ApiSchema):
    """What the chatbot says. Written to `draft_config["ai"]`.

    Caps match `helpers/prompt_builder.py`, which is where these values are
    spent: a persona long enough to crowd out the website knowledge is not a
    customisation, it is a regression in answer quality.
    """

    chatbot_name: str | None = Field(default=None, max_length=120)
    business_name: str | None = Field(default=None, max_length=MAX_BUSINESS_NAME_CHARACTERS)
    greeting: str | None = Field(default=None, max_length=MAX_GREETING_CHARACTERS)
    welcome_message: str | None = Field(default=None, max_length=MAX_WELCOME_CHARACTERS)
    suggested_questions: list[str] | None = Field(
        default=None, max_length=MAX_SUGGESTED_QUESTIONS
    )

    persona: str | None = Field(default=None, max_length=400)
    instructions: str | None = Field(default=None, max_length=1_200)
    tone: str | None = Field(default=None, max_length=40)
    language: str | None = Field(default=None, max_length=40)
    response_length: str | None = Field(default=None, max_length=20)
    fallback_message: str | None = Field(default=None, max_length=400)
    cite_sources: bool | None = None
    grounding: str | None = Field(default=None, max_length=20)

    @field_validator(
        "chatbot_name",
        "business_name",
        "greeting",
        "welcome_message",
        "persona",
        "instructions",
        "fallback_message",
    )
    @classmethod
    def _trim(cls, value: str | None) -> str | None:
        return _clean_text(value)

    @field_validator("suggested_questions")
    @classmethod
    def _clean_questions(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned: list[str] = []
        for raw in value:
            if not isinstance(raw, str):
                raise ValueError("Each suggested question must be text.")
            question = raw.strip()[:MAX_SUGGESTED_QUESTION_CHARACTERS]
            if question and question not in cleaned:
                cleaned.append(question)
        return cleaned


class ColorOverridesInput(ApiSchema):
    """Explicit colour choices. Anything left out is derived from the primary.

    Every field is optional and nullable, and that is load-bearing: a customer
    who sets only the primary colour gets a coherent palette derived from it,
    and clearing one colour returns it to being derived rather than leaving a
    hole.
    """

    primary: str | None = None
    secondary: str | None = None
    header_background: str | None = None
    header_text: str | None = None
    chat_background: str | None = None
    user_bubble: str | None = None
    user_bubble_text: str | None = None
    ai_bubble: str | None = None
    message_text: str | None = None

    @field_validator("*")
    @classmethod
    def _check_color(cls, value: str | None) -> str | None:
        if value is None:
            return None
        candidate = value.strip()
        if not candidate:
            return None
        if not is_hex_color(candidate):
            raise ValueError(
                f"'{candidate[:20]}' is not a colour. Use a hex value such as #7d2a47."
            )
        return candidate.lower()


class AppearanceSettingsInput(ApiSchema):
    """How the chat window looks. Written to `draft_config["appearance"]`."""

    theme: str | None = Field(default=None, max_length=20)
    font: str | None = Field(default=None, max_length=30)
    border_radius: int | None = Field(
        default=None, ge=MIN_BORDER_RADIUS, le=MAX_BORDER_RADIUS
    )
    # `asset:<uuid>` for an upload, or an absolute https URL. Null clears.
    avatar: str | None = Field(default=None, max_length=2048)
    logo: str | None = Field(default=None, max_length=2048)
    colors: ColorOverridesInput | None = None


class LauncherSettingsInput(ApiSchema):
    """The floating button. Written to `chatbots.deployment_config`.

    Ranges are the same ones `LauncherSettings` enforces on the way out, so a
    value that would produce a broken launcher is refused here with a message
    the customer can act on, rather than silently clamped later.
    """

    icon: str | None = Field(default=None, max_length=30)
    icon_image: str | None = Field(default=None, max_length=2048)
    accent_color: str | None = None
    launcher_size: int | None = Field(default=None, ge=40, le=96)
    shape: str | None = Field(default=None, max_length=20)
    position: str | None = Field(default=None, max_length=20)
    offset_x: int | None = Field(default=None, ge=0, le=200)
    offset_y: int | None = Field(default=None, ge=0, le=200)
    window_width: int | None = Field(default=None, ge=280, le=640)
    window_height: int | None = Field(default=None, ge=360, le=900)
    label: str | None = Field(default=None, max_length=MAX_LAUNCHER_LABEL_CHARACTERS)
    open_animation: str | None = Field(default=None, max_length=20)
    close_animation: str | None = Field(default=None, max_length=20)
    attention_animation: str | None = Field(default=None, max_length=20)
    open_on_load: bool | None = None
    show_branding: bool | None = None

    @field_validator("accent_color")
    @classmethod
    def _check_accent(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        if not is_hex_color(value):
            raise ValueError("The launcher colour must be a hex value such as #7d2a47.")
        return value.strip().lower()

    @field_validator("label")
    @classmethod
    def _trim_label(cls, value: str | None) -> str | None:
        return _clean_text(value)


class CustomizationUpdate(ApiSchema):
    """The studio's save request. Only the groups sent are written."""

    content: ContentSettingsInput | None = None
    appearance: AppearanceSettingsInput | None = None
    launcher: LauncherSettingsInput | None = None


class StudioChatRequest(ChatRequest):
    """Ask a real question using the owner's current, unsaved design."""

    customization: CustomizationUpdate = Field(default_factory=CustomizationUpdate)


# ---------------------------------------------------------------------------
# Response: resolved settings
# ---------------------------------------------------------------------------


class LauncherSettings(ApiSchema):
    """Resolved floating-launcher settings.

    Every value has a default in `config.py`, so a chatbot whose
    `deployment_config` is an empty object still renders a correct launcher.
    Phase 3 introduced this class as read-only; Phase 4's studio writes it
    through `LauncherSettingsInput`, which enforces the same ranges on the way
    in. The ranges are not cosmetic: `helpers/widget.py` interpolates these
    numbers into a script that runs on the customer's own website, and an
    unbounded window size there would cover their page.
    """

    icon: str = "chat"
    # A custom launcher image: an absolute https URL, resolved from an
    # `asset:<uuid>` reference when the customer uploaded one.
    icon_url: str | None = None
    accent_color: str = "#7d2a47"
    launcher_size: int = Field(default=56, ge=40, le=96)
    shape: str = "circle"
    position: str = "bottom-right"
    offset_x: int = Field(default=20, ge=0, le=200)
    offset_y: int = Field(default=20, ge=0, le=200)
    window_width: int = Field(default=400, ge=280, le=640)
    window_height: int = Field(default=600, ge=360, le=900)
    # A short word or two beside the button, e.g. "Chat with us".
    label: str | None = None
    open_animation: str = "scale"
    close_animation: str = "scale"
    attention_animation: str = "none"
    open_on_load: bool = False
    show_branding: bool = True


class ResolvedContent(ApiSchema):
    """Content settings with every default applied."""

    chatbot_name: str
    business_name: str | None = None
    greeting: str
    welcome_message: str | None = None
    suggested_questions: list[str] = Field(default_factory=list)
    persona: str
    instructions: str
    tone: str
    language: str
    response_length: str
    fallback_message: str
    cite_sources: bool = True
    grounding: str


class Palette(ApiSchema):
    """Every colour the chat UI paints with, for one theme.

    `border` and `muted_text` are derived rather than customer-settable: they
    exist so the component never has to invent a colour, and deriving them
    from the chosen background is what keeps a dark theme readable without
    asking the customer to pick eleven colours.
    """

    primary: str
    secondary: str
    header_background: str
    header_text: str
    chat_background: str
    user_bubble: str
    user_bubble_text: str
    ai_bubble: str
    message_text: str
    border: str
    muted_text: str


class ResolvedAppearance(ApiSchema):
    """Looks, resolved for both themes.

    Both palettes are always sent. `theme: "auto"` is then resolved in the
    browser from `prefers-color-scheme`, which is the only place that answer
    exists — a server has no way to know, and sending one palette would make
    "automatic" a lie that happened to match half the time.
    """

    theme: str
    font: str
    font_stack: str
    border_radius: int
    avatar_url: str | None = None
    logo_url: str | None = None
    light: Palette
    dark: Palette


class CustomizationState(ApiSchema):
    """One complete, resolved configuration — a draft or a published snapshot."""

    content: ResolvedContent
    appearance: ResolvedAppearance
    launcher: LauncherSettings


class AssetOut(ApiSchema):
    """An uploaded image, as the studio sees it."""

    id: uuid.UUID
    kind: str
    content_type: str
    byte_size: int
    created_at: datetime
    # The absolute URL the chat UI loads, built from PUBLIC_API_BASE_URL.
    url: str
    # What to store in the configuration to use this image.
    reference: str


class AssetUploadRequest(ApiSchema):
    """An image upload, as base64 or a data URL.

    JSON rather than multipart on purpose: multipart would add
    `python-multipart` to the dependency list for one endpoint, and an avatar
    is small enough that the ~33% base64 overhead is irrelevant. The browser
    already has the bytes as a data URL from `FileReader`.

    The declared content type is *not* trusted — `helpers/customization.py`
    verifies the magic bytes and stores what it found, not what was claimed.
    """

    kind: AssetKind
    data: str = Field(min_length=8, max_length=ASSET_MAX_BASE64_CHARACTERS)


class OptionItem(ApiSchema):
    """One selectable value, with the label the studio shows."""

    value: str
    label: str
    description: str | None = None
    # Fonts only: the CSS stack, so the picker can render each name in itself.
    font_stack: str | None = None
    # Colours only: the hex value a preset applies.
    color: str | None = None
    # Launcher icons only: the SVG path data from helpers/icons.py, so the
    # studio's gallery and its launcher preview draw exactly the glyph the
    # generated loader will draw on the customer's website.
    svg_path: str | None = None


class CustomizationLimits(ApiSchema):
    """The numbers the studio needs to constrain its own controls."""

    max_suggested_questions: int = MAX_SUGGESTED_QUESTIONS
    max_suggested_question_characters: int = MAX_SUGGESTED_QUESTION_CHARACTERS
    max_persona_characters: int = 400
    max_instructions_characters: int = 1_200
    max_fallback_characters: int = 400
    max_greeting_characters: int = MAX_GREETING_CHARACTERS
    max_welcome_characters: int = MAX_WELCOME_CHARACTERS
    max_launcher_label_characters: int = MAX_LAUNCHER_LABEL_CHARACTERS
    border_radius_range: list[int] = Field(
        default_factory=lambda: [MIN_BORDER_RADIUS, MAX_BORDER_RADIUS]
    )
    launcher_size_range: list[int] = Field(default_factory=lambda: [40, 96])
    launcher_offset_range: list[int] = Field(default_factory=lambda: [0, 200])
    window_width_range: list[int] = Field(default_factory=lambda: [280, 640])
    window_height_range: list[int] = Field(default_factory=lambda: [360, 900])
    asset_max_bytes: int = ASSET_MAX_BYTES
    asset_content_types: list[str] = Field(default_factory=lambda: list(ASSET_CONTENT_TYPES))


class CustomizationOptions(ApiSchema):
    """Every valid choice, served to the studio instead of hardcoded in it."""

    themes: list[OptionItem]
    fonts: list[OptionItem]
    tones: list[OptionItem]
    languages: list[OptionItem]
    response_lengths: list[OptionItem]
    grounding_levels: list[OptionItem]
    launcher_icons: list[OptionItem]
    launcher_positions: list[OptionItem]
    launcher_shapes: list[OptionItem]
    open_animations: list[OptionItem]
    close_animations: list[OptionItem]
    attention_animations: list[OptionItem]
    color_presets: list[OptionItem]
    limits: CustomizationLimits = Field(default_factory=CustomizationLimits)


class CustomizationOut(ApiSchema):
    """Everything the studio renders."""

    chatbot_id: uuid.UUID
    public_id: str
    status: ChatbotStatus
    is_published: bool
    can_publish: bool = False
    publish_blocked_reason: str | None = None
    draft: CustomizationState
    # Absent until the chatbot has been published at least once.
    published: CustomizationState | None = None
    # True when the draft differs from what visitors are being served, so the
    # studio can say "publish to apply" rather than leaving the customer
    # wondering why their live chatbot still looks the same.
    has_unpublished_changes: bool = False
    # Setting paths the customer has explicitly chosen, e.g.
    # "appearance.colors.primary". Lets the studio distinguish "derived from
    # your brand colour" from "you set this", and offer to clear it.
    overrides: list[str] = Field(default_factory=list)
    assets: list[AssetOut] = Field(default_factory=list)
    options: CustomizationOptions

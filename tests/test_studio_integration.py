"""Regression checks for the studio's save/preview/publish boundaries."""

from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest

from helpers import customization, deployment, errors, widget
from helpers.chat_engine import TurnRequest
from models.conversation import ConversationSource
from models.customization import CustomizationUpdate, StudioChatRequest
from routers import customization as routes
from tests.test_customization import make_chatbot


async def test_unsaved_preview_does_not_mutate_the_saved_design(monkeypatch):
    bot = make_chatbot(draft={"ai": {"tone": "formal"}})
    db = AsyncMock()
    monkeypatch.setattr(customization, "get_chatbot", AsyncMock(return_value=bot))
    shadow = await customization.preview_chatbot(
        db, SimpleNamespace(), bot.id,
        CustomizationUpdate(content={"tone": "friendly", "instructions": "Offer a callback."}),
    )
    assert deployment.ai_config_for(shadow, published=False).tone == "friendly"
    assert bot.draft_config == {"ai": {"tone": "formal"}}
    db.commit.assert_not_awaited()


async def test_unsaved_personality_reaches_the_actual_chat_pipeline(monkeypatch):
    bot = make_chatbot()
    shadow = make_chatbot(draft={"ai": {"tone": "formal", "language": "Hindi"}})
    request = TurnRequest(chatbot=bot, config=deployment.ai_config_for(bot, published=False),
        question="What are your hours?", source=ConversationSource.PLAYGROUND)
    monkeypatch.setattr(customization, "preview_chatbot", AsyncMock(return_value=shadow))
    monkeypatch.setattr(routes.chat_engine, "resolve_playground_turn", AsyncMock(return_value=request))
    prepare = AsyncMock(return_value=SimpleNamespace())
    monkeypatch.setattr(routes.chat_engine, "prepare", prepare)
    monkeypatch.setattr(routes.chat_engine, "complete", AsyncMock(return_value="answer"))
    payload = StudioChatRequest(message="What are your hours?", customization={"content": {"tone": "formal"}})
    assert await routes.preview_chat(bot.id, payload, SimpleNamespace(), AsyncMock(), False) == "answer"
    prepared_request = prepare.call_args.args[1]
    assert prepared_request.config.tone == "formal"
    assert prepared_request.config.language == "Hindi"
    assert prepared_request.chatbot is shadow
    prepare.assert_awaited_once()


async def test_studio_cannot_preview_someone_elses_chatbot(monkeypatch):
    monkeypatch.setattr(customization, "get_chatbot", AsyncMock(side_effect=errors.not_found("chatbot")))
    with pytest.raises(errors.AppError) as caught:
        await customization.preview_chatbot(AsyncMock(), SimpleNamespace(), make_chatbot().id, CustomizationUpdate())
    assert caught.value.status_code == 404


def test_closing_animation_survives_draft_and_published_resolution():
    bot = make_chatbot(launcher={"close_animation": "fade"}, published={"deployment": {"close_animation": "slide"}})
    assert customization.resolve_launcher(bot).close_animation == "fade"
    assert customization.resolve_launcher(bot, published=True).close_animation == "slide"


def test_preview_and_deployment_share_the_same_widget_runtime():
    bot = make_chatbot()
    launcher = customization.resolve_launcher(bot)
    preview = widget.build_loader_script(bot, launcher, preview=True)
    deployed = widget.build_loader_script(bot, launcher)
    # Only resolved configuration differs; the implementation is identical.
    preview_runtime = preview[preview.index("  // Guard:"):]
    deployed_runtime = deployed[deployed.index("  // Guard:"):]
    assert preview_runtime == deployed_runtime
    assert widget.build_loader_config(bot, launcher, preview=True)["url"] == "about:blank"
    assert widget.build_loader_config(bot, launcher)["refresh"].endswith("/widget-config")


def test_custom_launcher_url_cannot_inject_image_attributes():
    from models.customization import LauncherSettings
    icon = widget._icon_markup(LauncherSettings(icon_url='https://example.com/x" onerror="alert(1)'))
    assert 'src="https://example.com/x&quot; onerror=&quot;alert(1)"' in icon
    assert ' onerror="' not in icon

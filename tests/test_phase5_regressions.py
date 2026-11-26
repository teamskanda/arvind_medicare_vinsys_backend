from types import SimpleNamespace
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock
import uuid
import pytest
from pydantic import ValidationError
from config import Settings, get_settings
from helpers import deployment, retrieval, prompt_builder, token_manager, crawling, errors
from models.chatbot import Chatbot

def test_generic_store_overview_uses_real_homepage_content():
    page = retrieval.PageRow(id=uuid.uuid4(),url='https://books.test/',title='Catalogue',content='# Books\n\nA Light in the Attic is available for GBP 51.77. A selection of books and reading material.',priority=100)
    result=retrieval.search([page]*20,'What does this store sell?',budget_tokens=80,max_sections=2,min_score=.08)
    assert result.sections
    assert 'A Light in the Attic' in result.sections[0].text

def test_unrelated_questions_do_not_trigger_overview_fallback():
    assert not retrieval.needs_overview('Give me a cake recipe.')

def test_reading_material_paraphrase_matches_books():
    assert 'book' in retrieval.extract_keywords('What reading material can I purchase here?')

def test_followup_retrieval_uses_only_last_user_question():
    query=retrieval.contextual_query('How much does that book cost?',[('user','What is A Light in the Attic?'),('assistant','A book from the catalogue.')])
    assert 'A Light in the Attic' in query
    assert 'A book from the catalogue' not in query
    assert retrieval.contextual_query('What is your contact number?',[('user','Previous unrelated question')])=='What is your contact number?'

def test_max_custom_instructions_preserve_language_and_grounding():
    config=prompt_builder.ChatbotAiConfig(chatbot_name='Test assistant',instructions='Business guidance. '*60,persona='Helpful guide. '*25,language='Hindi',response_length='short',grounding='strict')
    prompt=prompt_builder.build_prompt(config=config,sections=[],question='Hello',history=[],budget=token_manager.build_budget(),system_role_mode='system')
    assert 'Reply in Hindi.' in prompt.system_text
    assert 'State only what the website content says.' in prompt.system_text
    assert 'same language' not in prompt.system_text

def test_browser_frame_policy_uses_validated_domains_and_allows_standalone():
    bot=Chatbot(allowed_domains=['example.com','*.example.org'])
    policy=deployment.frame_ancestors(bot)
    assert "'self'" in policy
    assert 'https://example.com' in policy
    assert 'https://*.example.org' in policy
    assert deployment.frame_ancestors(Chatbot(allowed_domains=[]))=='*'
    assert deployment.frame_ancestors(Chatbot(allowed_domains=['example.com; script-src *']))=="'none'"

def test_production_rejects_localhost_or_missing_visitor_secret():
    with pytest.raises(ValidationError):
        Settings(database_url=get_settings().database_url,environment='production')
    valid=dict(database_url=get_settings().database_url,environment='production',public_api_base_url='https://api.skanda.test',public_widget_base_url='https://chat.skanda.test',frontend_url='https://chat.skanda.test',visitor_session_secret='a'*48,account_auth_secret='b'*48)
    assert Settings(**valid).environment=='production'
    with pytest.raises(ValidationError):
        Settings(**{**valid,'public_api_base_url':'http://api.skanda.test'})

async def test_failed_poll_preserves_successful_pages(monkeypatch):
    db=SimpleNamespace(execute=AsyncMock(),commit=AsyncMock())
    monkeypatch.setattr(crawling,'_page_counts',AsyncMock(return_value={'scraped':2,'duplicate':1,'empty':0}))
    setter=AsyncMock()
    monkeypatch.setattr(crawling.chatbot_helpers,'set_chatbot_crawl_status',setter)
    await crawling._mark_failed(db,SimpleNamespace(id=uuid.uuid4(),chatbot_id=uuid.uuid4()),'Provider timeout')
    assert setter.call_args.args[-1].value=='partially_completed'

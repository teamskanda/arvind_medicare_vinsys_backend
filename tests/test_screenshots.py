"""Durable screenshots: bounded bytes, SSRF, ownership and race protection."""
import base64
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock
import uuid
import httpx
from PIL import Image
import pytest
from helpers import screenshots, firecrawl, errors, crawling
from models.screenshot import MAX_SCREENSHOT_BYTES

def png():
    out = BytesIO()
    Image.new('RGB', (100, 80), '#123456').save(out, 'PNG')
    return out.getvalue()

def test_screenshots_are_fully_decoded_and_reencoded():
    data = screenshots.normalize_image(png())
    with Image.open(BytesIO(data)) as image:
        assert image.format == 'JPEG'
        assert image.size == (100, 80)

@pytest.mark.parametrize('data', [b'', b'<svg onload="alert(1)"></svg>', b'\x89PNG\r\n\x1a\nnot-an-image', b'x' * (MAX_SCREENSHOT_BYTES + 1)], ids=['empty','svg','truncated-png','oversize'])
def test_invalid_or_oversized_images_rejected(data):
    with pytest.raises(errors.AppError) as caught:
        screenshots.normalize_image(data)
    assert caught.value.status_code == 422

def test_excessive_dimensions_rejected():
    out=BytesIO()
    Image.new('1', (5000, 4000)).save(out, 'PNG')
    with pytest.raises(errors.AppError):
        screenshots.normalize_image(out.getvalue())

def test_capture_only_homepage_without_customer_query():
    assert screenshots.homepage_url('https://acme.com/products?a=1#x') == 'https://acme.com/'

def test_interrupted_capture_is_recoverable():
    row=SimpleNamespace(status='capturing', updated_at=datetime.now(timezone.utc)-timedelta(minutes=3))
    assert screenshots.state(row).status == 'unavailable'

@pytest.mark.parametrize('url', ['https://127.0.0.1/img', 'https://localhost/img', 'http://example.com/img'])
async def test_unsafe_screenshot_downloads_rejected(url, monkeypatch):
    if url.startswith('http:'):
        monkeypatch.setattr(screenshots.url_validation, 'validate_url', AsyncMock(return_value=SimpleNamespace(scheme='http')))
    with pytest.raises((errors.AppError, ValueError)):
        await screenshots.download_image(url)

async def test_missing_provider_screenshot_is_optional(monkeypatch):
    monkeypatch.setattr(firecrawl,'_client',lambda: httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(200,json={'success':True,'data':{}}))))
    with pytest.raises(ValueError):
        await firecrawl.capture_screenshot('https://example.com/')

async def test_provider_only_captures_one_homepage(monkeypatch):
    import json
    seen=[]
    def handle(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200,json={'data':{'screenshot':'https://storage.example.com/signed-image'}})
    monkeypatch.setattr(firecrawl,'_client',lambda: httpx.AsyncClient(transport=httpx.MockTransport(handle)))
    assert await firecrawl.capture_screenshot('https://example.com/') == 'https://storage.example.com/signed-image'
    assert seen[0]['formats'][0]['type'] == 'screenshot'
    assert seen[0]['formats'][0]['fullPage'] is False
    assert seen[0]['maxAge'] > 0
    assert firecrawl.build_crawl_payload('https://example.com/',limit=20,max_depth=3)['scrapeOptions']['formats'] == ['markdown']

async def test_capture_failure_preserves_crawl_and_checks_version(monkeypatch):
    db=SimpleNamespace(execute=AsyncMock(),commit=AsyncMock())
    @asynccontextmanager
    async def scope(): yield db
    monkeypatch.setattr(screenshots,'session_scope',scope)
    monkeypatch.setattr(firecrawl,'capture_screenshot',AsyncMock(side_effect=ValueError('no screenshot')))
    bot_id,owner_id,version=uuid.uuid4(),uuid.uuid4(),uuid.uuid4()
    await screenshots._capture(bot_id,owner_id,version,'https://example.com/')
    statement=db.execute.call_args.args[0]
    compiled=statement.compile()
    assert 'chatbot_screenshots.version =' in str(statement)
    assert 'chatbot_screenshots.owner_id =' in str(statement)
    assert compiled.params['status'] == 'unavailable'
    assert 'chatbots SET' not in str(statement)
    db.commit.assert_awaited_once()

async def test_ready_or_manually_uploaded_screenshot_never_auto_recaptured(monkeypatch):
    row=SimpleNamespace(status='ready',source='manual',updated_at=datetime.now(timezone.utc),chatbot_id=uuid.uuid4(),version=uuid.uuid4(),message=None)
    monkeypatch.setattr(screenshots,'get_chatbot',AsyncMock())
    monkeypatch.setattr(screenshots,'read',AsyncMock(return_value=row))
    db=SimpleNamespace(execute=AsyncMock())
    result=await screenshots.schedule(db,SimpleNamespace(id=uuid.uuid4()),row.chatbot_id,retry=True)
    assert result.status == 'ready'
    db.execute.assert_not_awaited()

async def test_screenshot_owner_checked_before_upload_validation(monkeypatch):
    monkeypatch.setattr(screenshots,'get_chatbot',AsyncMock(side_effect=errors.not_found()))
    with pytest.raises(errors.AppError) as caught:
        await screenshots.upload(None,None,uuid.uuid4(),'junk')
    assert caught.value.status_code == 404

async def test_retryable_provider_error_cannot_run_forever(monkeypatch):
    from config import get_settings
    monkeypatch.setattr(crawling,'get_settings',lambda:get_settings().model_copy(update={'crawl_max_duration_seconds':0}))
    db=SimpleNamespace()
    @asynccontextmanager
    async def scope(): yield db
    monkeypatch.setattr(crawling,'session_scope',scope)
    monkeypatch.setattr(crawling,'_load_job',AsyncMock(return_value=SimpleNamespace(status='crawling')))
    monkeypatch.setattr(crawling,'sync_job',AsyncMock(side_effect=errors.provider_error('Timeout')))
    fail=AsyncMock()
    monkeypatch.setattr(crawling,'_fail_job',fail)
    bot,job=uuid.uuid4(),uuid.uuid4()
    await crawling._poll_until_done(bot,job)
    fail.assert_awaited_once_with(job,bot,'Timeout')

async def test_saved_pages_survive_provider_failure(monkeypatch):
    monkeypatch.setattr(crawling,'_page_counts',AsyncMock(return_value={'scraped':3,'duplicate':1}))
    status=AsyncMock()
    monkeypatch.setattr(crawling.chatbot_helpers,'set_chatbot_crawl_status',status)
    db=SimpleNamespace(execute=AsyncMock(),commit=AsyncMock())
    bot=uuid.uuid4()
    await crawling._mark_failed(db,SimpleNamespace(id=uuid.uuid4(),chatbot_id=bot),'Timeout')
    assert status.call_args.args[-1].value == 'partially_completed'
    params=db.execute.call_args.args[0].compile().params
    assert params['successful_pages'] == 3
    assert 'saved before scraping stopped' in params['error_details']['message']

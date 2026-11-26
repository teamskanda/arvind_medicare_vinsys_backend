"""Provider failures are simulated deliberately, without burning real quota."""
from types import SimpleNamespace
from unittest.mock import AsyncMock
import uuid
import httpx
import pytest
from helpers import crawling, errors, firecrawl

@pytest.mark.parametrize('status',[401,402,403,404,429,500])
def test_provider_failures_are_customer_readable(status):
    response=httpx.Response(status,json={'error':'Provider rejected the request'})
    with pytest.raises(errors.AppError) as caught:
        firecrawl._raise_for_provider_status(response)
    assert caught.value.status_code >= 400
    assert caught.value.message
    if status==402:
        assert caught.value.code=='provider_out_of_credits'
        assert caught.value.retryable is False

async def test_start_timeout_is_a_recoverable_error(monkeypatch):
    def timeout(request): raise httpx.ReadTimeout('Timeout',request=request)
    monkeypatch.setattr(firecrawl,'_client',lambda:httpx.AsyncClient(transport=httpx.MockTransport(timeout)))
    with pytest.raises(errors.AppError) as caught:
        await firecrawl.start_crawl('https://example.com/')
    assert caught.value.retryable
    assert 'time' in caught.value.message

async def test_malformed_provider_output_never_becomes_a_successful_job(monkeypatch):
    transport=httpx.MockTransport(lambda request:httpx.Response(200,text='not JSON'))
    monkeypatch.setattr(firecrawl,'_client',lambda:httpx.AsyncClient(transport=transport))
    with pytest.raises(errors.AppError):
        await firecrawl.start_crawl('https://example.com/')

async def test_partial_success_and_robots_blocks_remain_visible(monkeypatch):
    db=SimpleNamespace(execute=AsyncMock(),commit=AsyncMock())
    job=SimpleNamespace(id=uuid.uuid4(),chatbot_id=uuid.uuid4(),provider_job_id='provider-job')
    report=firecrawl.CrawlErrorReport(failed_urls=[{'url':'https://example.com/blocked','error':'Blocked'}],robots_blocked=['https://example.com/private'])
    monkeypatch.setattr(firecrawl,'get_crawl_errors',AsyncMock(return_value=report))
    monkeypatch.setattr(crawling,'_page_counts',AsyncMock(return_value={'scraped':2,'duplicate':0,'empty':0}))
    setter=AsyncMock();monkeypatch.setattr(crawling.chatbot_helpers,'set_chatbot_crawl_status',setter)
    await crawling._finalise(db,job,firecrawl.CrawlSnapshot(provider_status='completed',total=3,completed=2),timed_out=False)
    params=db.execute.call_args.args[0].compile().params
    assert params['status']=='partially_completed'
    assert params['successful_pages']==2
    assert params['failed_pages']==1
    assert params['error_details']['robots_blocked']==['https://example.com/private']

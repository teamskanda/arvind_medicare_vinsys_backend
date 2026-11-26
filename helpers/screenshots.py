"""Optional homepage capture: isolated from crawling, durable, owner-scoped.

Automatic attempts occur once per bot. Explicit retries are allowed only when
there is no ready image. A version guard ensures a slow capture can never
overwrite a manual replacement or resurrect a removed image.
"""
from __future__ import annotations
import asyncio
import base64
import binascii
from datetime import datetime, timedelta, timezone
from io import BytesIO
import logging
from urllib.parse import urljoin, urlsplit, urlunsplit
import uuid
import warnings
import httpx
from PIL import Image, UnidentifiedImageError
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert
from helpers import errors, firecrawl, url_validation
from helpers.chatbots import get_chatbot
from helpers.database import session_scope
from models.screenshot import MAX_SCREENSHOT_BYTES, ScreenshotOut, WebsiteScreenshot

logger = logging.getLogger(__name__)
_tasks: set[asyncio.Task] = set()

def homepage_url(url: str) -> str:
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, '/', '', ''))

def normalize_image(data: bytes) -> bytes:
    """Decode fully, bound dimensions, strip metadata, and re-encode as JPEG."""
    if not data or len(data) > MAX_SCREENSHOT_BYTES:
        raise errors.AppError('invalid_screenshot', 'Choose an image under 5 MB.', status_code=422)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                if image.format not in {'PNG', 'JPEG', 'WEBP'} or image.width * image.height > 16_000_000:
                    raise ValueError('Unsupported image or dimensions')
                image.load()
                image.thumbnail((1920, 4000))
                output = BytesIO()
                image.convert('RGB').save(output, format='JPEG', quality=85, optimize=True)
        result = output.getvalue()
        if len(result) > MAX_SCREENSHOT_BYTES:
            raise ValueError('Image too large')
        return result
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise errors.AppError('invalid_screenshot', 'That screenshot is not a valid PNG, JPG or WebP image, or its dimensions are too large.', status_code=422) from None

async def download_image(url: str) -> bytes:
    """No provider credentials on image downloads. Pin DNS on every redirect."""
    async with asyncio.timeout(35):
        async with httpx.AsyncClient(timeout=15, follow_redirects=False) as client:
            for _ in range(4):
                validated = await url_validation.validate_url(url)
                if validated.scheme != 'https':
                    raise ValueError('Screenshot storage must use HTTPS')
                async with client.stream('GET', url_validation._pinned_target(validated, validated.resolved_ips[0]),
                    headers={'Host': validated.host}, extensions={'sni_hostname': validated.host}) as response:
                    if response.is_redirect:
                        url = urljoin(url, response.headers.get('location', ''))
                        continue
                    response.raise_for_status()
                    data = bytearray()
                    async for chunk in response.aiter_bytes():
                        data.extend(chunk)
                        if len(data) > MAX_SCREENSHOT_BYTES:
                            raise ValueError('Screenshot exceeds 5 MB')
                    return bytes(data)
    raise ValueError('Screenshot redirects exceeded')

def state(row: WebsiteScreenshot | None) -> ScreenshotOut:
    if row is None:
        return ScreenshotOut(status='unavailable', message='No website screenshot yet. You can upload one.')
    if row.status == 'capturing' and row.updated_at < datetime.now(timezone.utc) - timedelta(minutes=2):
        return ScreenshotOut(status='unavailable', message='Screenshot capture was interrupted. You can retry or upload an image.')
    return ScreenshotOut(status=row.status, source=row.source, message=row.message,
        image_url=f'/api/chatbots/{row.chatbot_id}/screenshot/image?v={row.version}' if row.status == 'ready' else None)

async def read(db, user, chatbot_id):
    await get_chatbot(db, user, chatbot_id)
    return (await db.execute(select(WebsiteScreenshot).where(WebsiteScreenshot.chatbot_id == chatbot_id, WebsiteScreenshot.owner_id == user.id))).scalar_one_or_none()

async def schedule(db, user, chatbot_id, *, retry=False):
    bot = await get_chatbot(db, user, chatbot_id)
    existing = await read(db, user, chatbot_id)
    if existing:
        stale = existing.updated_at < datetime.now(timezone.utc) - timedelta(minutes=2)
        if existing.status == 'ready' or (existing.status == 'capturing' and not stale) or not retry:
            return state(existing)
    version = uuid.uuid4()
    values = dict(chatbot_id=bot.id, owner_id=user.id, version=version, status='capturing', source='automatic',
        source_url=homepage_url(bot.website_url), message=None, data=None, content_type=None, updated_at=datetime.now(timezone.utc))
    stmt = insert(WebsiteScreenshot).values(**values)
    if existing:
        # Concurrent clicks/requests cannot spend duplicate capture credits.
        stmt = stmt.on_conflict_do_update(index_elements=['chatbot_id'], set_=values,
            where=WebsiteScreenshot.version == existing.version)
    else:
        stmt = stmt.on_conflict_do_nothing(index_elements=['chatbot_id'])
    claimed = (await db.execute(stmt.returning(WebsiteScreenshot.version))).scalar_one_or_none()
    await db.commit()
    if claimed:
        task = asyncio.create_task(_capture(bot.id, user.id, version, values['source_url']))
        _tasks.add(task)
        task.add_done_callback(_tasks.discard)
    return ScreenshotOut(status='capturing', source='automatic', message='Capturing the homepage. Your chatbot is ready to use meanwhile.')

def on_chatbot_created(chatbot_id, user):
    async def start():
        try:
            async with session_scope() as db:
                await schedule(db, user, chatbot_id)
        except Exception:
            logger.warning('Optional screenshot could not start for %s.', chatbot_id, exc_info=True)
    task = asyncio.create_task(start())
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)

async def _capture(chatbot_id, owner_id, version, url):
    try:
        async with asyncio.timeout(80):
            provider_url = await firecrawl.capture_screenshot(url)
            image = await download_image(provider_url)
            image = await asyncio.to_thread(normalize_image, image)
        values = dict(status='ready', data=image, content_type='image/jpeg', message=None)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.info('Optional homepage capture unavailable for %s (%s)', chatbot_id, type(exc).__name__)
        values = dict(status='unavailable', message='The homepage screenshot was unavailable. Your chatbot and knowledge base are unaffected. Upload an image or retry capture.')
    try:
        async with session_scope() as db:
            await db.execute(update(WebsiteScreenshot).where(WebsiteScreenshot.chatbot_id == chatbot_id,
                WebsiteScreenshot.owner_id == owner_id, WebsiteScreenshot.version == version).values(**values, updated_at=datetime.now(timezone.utc)))
            await db.commit()
    except Exception:
        logger.warning('Could not save optional screenshot for %s.', chatbot_id, exc_info=True)

async def upload(db, user, chatbot_id, payload):
    bot = await get_chatbot(db, user, chatbot_id)
    try:
        prefix, encoded = payload.split(',', 1)
        if prefix not in {'data:image/png;base64', 'data:image/jpeg;base64', 'data:image/webp;base64'}:
            raise ValueError('Invalid data URL')
        image = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise errors.AppError('invalid_screenshot', 'Choose a PNG, JPG or WebP screenshot.', status_code=422) from None
    image = await asyncio.to_thread(normalize_image, image)
    values = dict(chatbot_id=bot.id, owner_id=user.id, version=uuid.uuid4(), status='ready', source='manual',
        source_url=None, message=None, content_type='image/jpeg', data=image, updated_at=datetime.now(timezone.utc))
    await db.execute(insert(WebsiteScreenshot).values(**values).on_conflict_do_update(index_elements=['chatbot_id'], set_=values))
    await db.commit()
    return state((await db.execute(select(WebsiteScreenshot).where(WebsiteScreenshot.chatbot_id == bot.id).execution_options(populate_existing=True))).scalar_one())

async def remove(db, user, chatbot_id):
    bot = await get_chatbot(db, user, chatbot_id)
    values = dict(chatbot_id=bot.id, owner_id=user.id, version=uuid.uuid4(), status='removed', source='manual',
        data=None, content_type=None, source_url=None, message='Backdrop removed.', updated_at=datetime.now(timezone.utc))
    await db.execute(insert(WebsiteScreenshot).values(**values).on_conflict_do_update(index_elements=['chatbot_id'], set_=values))
    await db.commit()
    return ScreenshotOut(status='removed', message='Backdrop removed.')

"""Private website preview screenshots; never public widget assets."""
import uuid
from typing import Annotated
from fastapi import APIRouter, Depends, Response
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from helpers import screenshots, errors
from helpers.auth import CurrentUser
from helpers.database import get_db
from models.screenshot import ScreenshotOut, ScreenshotUpload, WebsiteScreenshot

router = APIRouter(prefix='/api/chatbots', tags=['screenshots'])
DbSession = Annotated[AsyncSession, Depends(get_db)]

@router.get('/{chatbot_id}/screenshot', response_model=ScreenshotOut)
async def read_screenshot(chatbot_id: uuid.UUID, user: CurrentUser, db: DbSession):
    return screenshots.state(await screenshots.read(db, user, chatbot_id))

@router.post('/{chatbot_id}/screenshot/capture', response_model=ScreenshotOut)
async def capture_screenshot(chatbot_id: uuid.UUID, user: CurrentUser, db: DbSession):
    return await screenshots.schedule(db, user, chatbot_id, retry=True)

@router.put('/{chatbot_id}/screenshot', response_model=ScreenshotOut)
async def upload_screenshot(chatbot_id: uuid.UUID, payload: ScreenshotUpload, user: CurrentUser, db: DbSession):
    return await screenshots.upload(db, user, chatbot_id, payload.data)

@router.delete('/{chatbot_id}/screenshot', response_model=ScreenshotOut)
async def remove_screenshot(chatbot_id: uuid.UUID, user: CurrentUser, db: DbSession):
    return await screenshots.remove(db, user, chatbot_id)

@router.get('/{chatbot_id}/screenshot/image')
async def screenshot_image(chatbot_id: uuid.UUID, user: CurrentUser, db: DbSession):
    row = await screenshots.read(db, user, chatbot_id)
    if row is None or row.status != 'ready':
        raise errors.not_found('Screenshot')
    data = (await db.execute(select(WebsiteScreenshot.data).where(WebsiteScreenshot.chatbot_id == chatbot_id, WebsiteScreenshot.owner_id == user.id))).scalar_one()
    return Response(content=data, media_type='image/jpeg', headers={'Cache-Control':'private, no-store', 'X-Content-Type-Options':'nosniff', 'Content-Security-Policy':"default-src 'none'; sandbox"})

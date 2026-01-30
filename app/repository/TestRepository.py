from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.database import get_db
from app.models.chat_history import ChatHistory


class TestRepository:
    def __init__(self, db: AsyncSession = Depends(get_db)):
        self.db = db

    async def create(self, session_id: str, sender: str, message: str):
        chat = ChatHistory(session_id=session_id, sender=sender, message=message)
        self.db.add(chat)
        await self.db.flush()
        return chat

    async def get_all(self, session_id: str):
        result = await self.db.execute(
            select(ChatHistory).where(ChatHistory.session_id == session_id)
        )
        return result.scalars().all()
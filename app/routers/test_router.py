from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from app.core.database import get_db
from app.repository.TestRepository import TestRepository


router = APIRouter(prefix="/test", tags=["Router Test"])

class ChatCreate(BaseModel):
    session_id: str
    sender: str
    message: str


@router.get("/check")
async def read_test():
    return {"message": "Test router is working!"}


@router.post("/chat")
async def create_chat(data: ChatCreate, db: AsyncSession = Depends(get_db)):
    repo = TestRepository(db)
    chat = await repo.create(data.session_id, data.sender, data.message)
    return {"id": chat.id, "message": chat.message}


@router.get("/chat/{session_id}")
async def get_chats(session_id: str, db: AsyncSession = Depends(get_db)):
    repo = TestRepository(db)
    chats = await repo.get_all(session_id)
    return [{"id": c.id, "sender": c.sender, "message": c.message} for c in chats]
# database.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import text
from dotenv import load_dotenv
import os

load_dotenv(override=True)
DATABASE_URL = os.getenv("DATABASE_URL")
IS_PRODUCTION = os.getenv("APP_ENV") == "production"
engine = create_async_engine(
    DATABASE_URL, 
    echo=False if IS_PRODUCTION else True,
    pool_size=20,
    max_overflow=10,
    pool_pre_ping=True,
    pool_recycle=3600,
)


AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# async def init_vector_extension():
#     """pgvector extension 초기화"""
#     async with engine.begin() as conn:
#         await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

from sqlalchemy import Column, Integer, String, Text
from pgvector.sqlalchemy import Vector
from app.core.database import Base


class DocumentEmbedding(Base):
    __tablename__ = "document_embeddings"

    id = Column(Integer, primary_key=True, index=True)
    content = Column(Text, nullable=False)
    source = Column(String(255), nullable=False)
    embedding = Column(Vector(1024), nullable=False)

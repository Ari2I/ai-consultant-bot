"""
Модели базы данных: документы базы знаний, их фрагменты (чанки) с
эмбеддингами и журнал вопросов/ответов клиентов.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
)


def _utcnow() -> datetime:
    """Возвращает текущее время в UTC как наивный datetime."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    """Базовый класс для всех моделей."""


class KnowledgeDocument(Base):
    """Документ базы знаний, загруженный админом (FAQ, прайс и т.д.)."""

    __tablename__ = "knowledge_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    admin_chat_id: Mapped[int] = mapped_column(Integer, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    original_format: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow
    )

    chunks: Mapped[List["KnowledgeChunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="KnowledgeChunk.chunk_index",
    )


class KnowledgeChunk(Base):
    """Один текстовый фрагмент документа вместе с его эмбеддингом."""

    __tablename__ = "knowledge_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_documents.id")
    )
    chunk_text: Mapped[str] = mapped_column(Text)
    # Эмбеддинг хранится как JSON-строка со списком float. Для
    # масштаба в десятки-сотни фрагментов (один небольшой бизнес)
    # поиск делается через numpy по всем векторам сразу (см. модуль
    # rag/search.py, который появится на следующем шаге) — полноценная
    # векторная БД (Qdrant/FAISS/Milvus) была бы избыточна. Для
    # значительно большего объёма документов такая БД потребовалась
    # бы — это ограничение явно зафиксировано в README проекта.
    embedding_json: Mapped[str] = mapped_column(Text)
    chunk_index: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow
    )

    document: Mapped["KnowledgeDocument"] = relationship(
        back_populates="chunks"
    )


class ConversationLog(Base):
    """Журнал одного вопроса клиента и ответа (или факта эскалации)."""

    __tablename__ = "conversation_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    customer_chat_id: Mapped[int] = mapped_column(Integer, index=True)
    question: Mapped[str] = mapped_column(Text)
    answer: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    similarity_score: Mapped[Optional[float]] = mapped_column(
        Float, nullable=True
    )
    # ID совпавших фрагментов через запятую — не строгая связь через
    # FK (фрагмент может быть позже удалён вместе с документом), а
    # диагностический след для /stats и разбора качества ответов.
    matched_chunk_ids: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )
    escalated: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, index=True
    )

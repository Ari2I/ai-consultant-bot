"""
Репозиторий — единая точка доступа к базе данных базы знаний и
журналу вопросов/ответов клиентов.

Изолирует остальной код приложения (обработчики бота, RAG-пайплайн)
от деталей работы с SQLAlchemy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker

from database.models import (
    Base,
    ConversationLog,
    KnowledgeChunk,
    KnowledgeDocument,
)


@dataclass
class ChunkRecord:
    """Фрагмент документа вместе с его эмбеддингом и именем файла."""

    id: int
    document_id: int
    filename: str
    chunk_text: str
    embedding: List[float]


@dataclass
class ConversationStats:
    """Простая аналитика по вопросам клиентов."""

    total_questions: int
    escalated_count: int
    recent_escalated: List[ConversationLog]


class KnowledgeRepository:
    """Репозиторий для базы знаний и журнала обращений клиентов."""

    def __init__(self, database_url: str) -> None:
        self._engine = create_engine(database_url)
        Base.metadata.create_all(self._engine)
        self._session_factory = sessionmaker(bind=self._engine)

    def _session(self) -> Session:
        return self._session_factory()

    # ------------------------------------------------------------------
    # База знаний: документы и чанки
    # ------------------------------------------------------------------
    def add_document(
        self, admin_chat_id: int, filename: str, original_format: str
    ) -> KnowledgeDocument:
        """Регистрирует новый документ базы знаний."""
        with self._session() as session:
            document = KnowledgeDocument(
                admin_chat_id=admin_chat_id,
                filename=filename,
                original_format=original_format,
            )
            session.add(document)
            session.commit()
            session.refresh(document)
            return document

    def add_chunks(
        self,
        document_id: int,
        chunks: Sequence[Tuple[str, List[float]]],
    ) -> None:
        """
        Сохраняет фрагменты документа вместе с их эмбеддингами.

        Аргументы:
            document_id: ID документа, которому принадлежат фрагменты.
            chunks: список пар (текст фрагмента, вектор эмбеддинга),
                в том порядке, в котором фрагменты идут в документе.
        """
        with self._session() as session:
            for index, (chunk_text, embedding) in enumerate(chunks):
                session.add(
                    KnowledgeChunk(
                        document_id=document_id,
                        chunk_text=chunk_text,
                        embedding_json=json.dumps(embedding),
                        chunk_index=index,
                    )
                )
            session.commit()

    def list_documents(self) -> List[KnowledgeDocument]:
        """Возвращает список всех загруженных документов."""
        with self._session() as session:
            stmt = select(KnowledgeDocument).order_by(
                KnowledgeDocument.created_at
            )
            return list(session.scalars(stmt).all())

    def remove_document(self, document_id: int) -> bool:
        """Удаляет документ и все его фрагменты. True, если найден."""
        with self._session() as session:
            document = session.get(KnowledgeDocument, document_id)
            if document is None:
                return False
            session.delete(document)
            session.commit()
            return True

    def get_all_chunks(self) -> List[ChunkRecord]:
        """
        Возвращает все фрагменты всех документов вместе с эмбеддингами.

        Используется поиском релевантного контекста (следующий модуль,
        rag/search.py) — для масштаба в десятки-сотни фрагментов
        загрузка всех векторов в память и сравнение через numpy
        эффективнее, чем обращение к полноценной векторной БД.
        """
        with self._session() as session:
            stmt = select(KnowledgeChunk, KnowledgeDocument.filename).join(
                KnowledgeDocument,
                KnowledgeChunk.document_id == KnowledgeDocument.id,
            )
            rows = session.execute(stmt).all()
            return [
                ChunkRecord(
                    id=chunk.id,
                    document_id=chunk.document_id,
                    filename=filename,
                    chunk_text=chunk.chunk_text,
                    embedding=json.loads(chunk.embedding_json),
                )
                for chunk, filename in rows
            ]

    # ------------------------------------------------------------------
    # Журнал вопросов клиентов
    # ------------------------------------------------------------------
    def log_conversation(
        self,
        customer_chat_id: int,
        question: str,
        answer: Optional[str],
        similarity_score: Optional[float],
        escalated: bool,
        matched_chunk_ids: Optional[List[int]] = None,
    ) -> ConversationLog:
        """Сохраняет запись о вопросе клиента и результате обработки."""
        with self._session() as session:
            log = ConversationLog(
                customer_chat_id=customer_chat_id,
                question=question,
                answer=answer,
                similarity_score=similarity_score,
                escalated=escalated,
                matched_chunk_ids=(
                    ",".join(str(i) for i in matched_chunk_ids)
                    if matched_chunk_ids
                    else None
                ),
            )
            session.add(log)
            session.commit()
            session.refresh(log)
            return log

    def get_stats(self, recent_limit: int = 10) -> ConversationStats:
        """Возвращает простую аналитику по обращениям клиентов."""
        with self._session() as session:
            total_questions = (
                session.scalar(select(func.count(ConversationLog.id)))
                or 0
            )
            escalated_count = (
                session.scalar(
                    select(func.count(ConversationLog.id)).where(
                        ConversationLog.escalated.is_(True)
                    )
                )
                or 0
            )

            recent_stmt = (
                select(ConversationLog)
                .where(ConversationLog.escalated.is_(True))
                .order_by(ConversationLog.created_at.desc())
                .limit(recent_limit)
            )
            recent_escalated = list(session.scalars(recent_stmt).all())

            return ConversationStats(
                total_questions=total_questions,
                escalated_count=escalated_count,
                recent_escalated=recent_escalated,
            )

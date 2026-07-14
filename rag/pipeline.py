"""
Оркестрация RAG-пайплайна: эмбеддинг вопроса клиента -> поиск
релевантных фрагментов базы знаний -> генерация ответа через GigaChat
или эскалация к админам, если уверенность найденного контекста ниже
порога.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from database.repository import KnowledgeRepository
from embeddings.encoder import EmbeddingEncoder
from llm.gigachat_client import GigaChatApiError, GigaChatClient
from rag.search import find_relevant_chunks


@dataclass
class AnswerResult:
    """Результат обработки одного вопроса клиента."""

    answer: Optional[str]
    escalated: bool
    similarity_score: Optional[float]
    matched_chunk_ids: List[int] = field(default_factory=list)
    error: Optional[str] = None


class RagPipeline:
    """Связывает поиск контекста и генерацию ответа в единый процесс."""

    def __init__(
        self,
        repository: KnowledgeRepository,
        encoder: EmbeddingEncoder,
        llm_client: GigaChatClient,
        similarity_threshold: float,
        top_k_chunks: int,
    ) -> None:
        self._repository = repository
        self._encoder = encoder
        self._llm_client = llm_client
        self._similarity_threshold = similarity_threshold
        self._top_k_chunks = top_k_chunks

    def answer_question(self, question: str) -> AnswerResult:
        """
        Обрабатывает один вопрос клиента от начала до конца.

        Логика эскалации:
            - база знаний пуста -> эскалация;
            - лучший найденный фрагмент по сходству ниже порога ->
              эскалация (модель не вызывается вообще — экономит
              токены на заведомо нерелевантном контексте);
            - GigaChat API вернул окончательную ошибку (после
              исчерпания ретраев) -> тоже эскалация, а не молчание,
              чтобы клиент в любом случае получил внимание человека.
        """
        chunks = self._repository.get_all_chunks()
        if not chunks:
            return AnswerResult(
                answer=None, escalated=True, similarity_score=None
            )

        query_embedding = self._encoder.encode(question)
        results = find_relevant_chunks(
            query_embedding, chunks, top_k=self._top_k_chunks
        )

        if not results or results[0].similarity < self._similarity_threshold:
            best_score = results[0].similarity if results else None
            return AnswerResult(
                answer=None,
                escalated=True,
                similarity_score=best_score,
                matched_chunk_ids=[r.chunk_id for r in results],
            )

        context = "\n\n".join(
            f"[{r.filename}] {r.chunk_text}" for r in results
        )
        matched_chunk_ids = [r.chunk_id for r in results]

        try:
            answer = self._llm_client.generate_answer(question, context)
        except GigaChatApiError as exc:
            return AnswerResult(
                answer=None,
                escalated=True,
                similarity_score=results[0].similarity,
                matched_chunk_ids=matched_chunk_ids,
                error=str(exc),
            )

        return AnswerResult(
            answer=answer,
            escalated=False,
            similarity_score=results[0].similarity,
            matched_chunk_ids=matched_chunk_ids,
        )

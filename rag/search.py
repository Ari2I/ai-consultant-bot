"""
Поиск наиболее релевантных фрагментов базы знаний по косинусному
сходству эмбеддингов.

Использует numpy для векторизованного вычисления сходства сразу со
всеми фрагментами. Для масштаба в десятки-сотни чанков (документы
одного небольшого бизнеса) это быстрее, чем сравнение по одному, и
не требует отдельной векторной БД (Qdrant/FAISS/Milvus) — см.
ограничения проекта в README. Для существенно большего объёма
документов такая БД потребовалась бы.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from database.repository import ChunkRecord


@dataclass
class SearchResult:
    """Один найденный релевантный фрагмент с оценкой сходства."""

    chunk_id: int
    document_id: int
    filename: str
    chunk_text: str
    similarity: float


def find_relevant_chunks(
    query_embedding: List[float],
    chunks: List[ChunkRecord],
    top_k: int,
) -> List[SearchResult]:
    """
    Возвращает top_k наиболее похожих на запрос фрагментов.

    Аргументы:
        query_embedding: эмбеддинг вопроса клиента.
        chunks: все доступные фрагменты базы знаний с их эмбеддингами.
        top_k: сколько наиболее релевантных фрагментов вернуть.

    Возвращает:
        Список SearchResult, отсортированный по убыванию сходства.
        Пустой список, если chunks пуст или эмбеддинг запроса нулевой.
    """
    if not chunks:
        return []

    query_vector = np.array(query_embedding, dtype=np.float64)
    query_norm = np.linalg.norm(query_vector)
    if query_norm == 0:
        return []

    matrix = np.array([chunk.embedding for chunk in chunks], dtype=np.float64)
    matrix_norms = np.linalg.norm(matrix, axis=1)

    # Защита от деления на ноль для нулевых векторов — в теории не
    # должно возникать для реальных эмбеддингов, но не должно и
    # приводить к NaN/падению, если вдруг возникнет.
    safe_norms = np.where(matrix_norms == 0, 1.0, matrix_norms)
    similarities = (matrix @ query_vector) / (safe_norms * query_norm)
    similarities = np.where(matrix_norms == 0, 0.0, similarities)

    limit = min(top_k, len(chunks))
    order = np.argsort(-similarities)[:limit]

    return [
        SearchResult(
            chunk_id=chunks[i].id,
            document_id=chunks[i].document_id,
            filename=chunks[i].filename,
            chunk_text=chunks[i].chunk_text,
            similarity=float(similarities[i]),
        )
        for i in order
    ]

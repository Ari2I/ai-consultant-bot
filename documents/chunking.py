"""
Разбиение извлечённого текста документа на фрагменты (чанки) для
последующего вычисления эмбеддингов и RAG-поиска.

Реализовано без внешних зависимостей (без LangChain, nltk и т.п.) —
согласно решению об архитектуре проекта: для текстов уровня
FAQ/прайс-листа небольшого бизнеса собственного разбиения по абзацам
и предложениям вполне достаточно, а лишняя зависимость не нужна.
"""

from __future__ import annotations

import re
from typing import List, Optional

# Разбиение предложений по знакам конца предложения с последующим
# пробелом/переносом строки. Не претендует на лингвистическую
# точность (например, не учитывает сокращения вида "т.д."), но для
# деления длинного абзаца на части этого достаточно.
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def _split_into_paragraphs(text: str) -> List[str]:
    """Делит текст на абзацы по пустым строкам между ними."""
    raw_paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in raw_paragraphs if p.strip()]


def _split_long_paragraph(paragraph: str, chunk_size: int) -> List[str]:
    """
    Делит один слишком длинный абзац на части по предложениям.

    Если даже одно предложение длиннее chunk_size (редкий крайний
    случай — например, текст без знаков препинания), оно жёстко
    режется по количеству символов, чтобы итоговый кусок не рос
    бесконечно.
    """
    sentences = _SENTENCE_SPLIT_PATTERN.split(paragraph)
    parts: List[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            parts.append(current)

        if len(sentence) <= chunk_size:
            current = sentence
        else:
            for start in range(0, len(sentence), chunk_size):
                parts.append(sentence[start:start + chunk_size])
            current = ""

    if current:
        parts.append(current)

    return parts


def split_text_into_chunks(
    text: str, chunk_size: int = 500, chunk_overlap: int = 50
) -> List[str]:
    """
    Разбивает текст на фрагменты заданного размера с перекрытием.

    Абзацы группируются в чанк, пока суммарная длина не превышает
    chunk_size. Между соседними чанками сохраняется перекрытие в
    chunk_overlap символов (конец предыдущего чанка становится
    началом следующего) — это снижает риск потери контекста на
    границе чанка при последующем поиске по эмбеддингам.

    Аргументы:
        text: исходный текст документа (результат documents.parsers).
        chunk_size: максимальный размер чанка в символах.
        chunk_overlap: размер перекрытия между соседними чанками в
            символах.

    Возвращает:
        Список текстовых фрагментов. Пустой список для пустого текста
        (например, если text состоит только из пробелов).

    Исключения:
        ValueError: если chunk_overlap >= chunk_size.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap должен быть меньше chunk_size "
            f"(получено chunk_size={chunk_size}, "
            f"chunk_overlap={chunk_overlap})"
        )

    paragraphs = _split_into_paragraphs(text)
    if not paragraphs:
        return []

    # Абзацы длиннее лимита разбиваем заранее по предложениям, чтобы
    # дальнейшая группировка ниже работала с гарантированно короткими
    # кусками текста.
    pieces: List[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            pieces.append(paragraph)
        else:
            pieces.extend(_split_long_paragraph(paragraph, chunk_size))

    chunks: List[str] = []
    current = ""

    for piece in pieces:
        candidate = f"{current}\n\n{piece}" if current else piece
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        chunks.append(current)
        overlap_tail: Optional[str] = (
            current[-chunk_overlap:] if chunk_overlap else None
        )
        current = f"{overlap_tail}\n\n{piece}" if overlap_tail else piece

    if current:
        chunks.append(current)

    return chunks

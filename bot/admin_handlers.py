"""
Обработчики команд администратора: управление базой знаний и простая
аналитика по обращениям клиентов.

Доступ ограничен списком ADMIN_CHAT_IDS из конфига — остальные
пользователи бота (клиенты) не видят и не могут вызвать эти команды,
а при попытке получают вежливый отказ (см. _is_admin).
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from aiogram import Bot, F, Router
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from bot.states import AddKnowledgeDocumentStates
from config import Settings
from database.repository import KnowledgeRepository
from documents.chunking import split_text_into_chunks
from documents.parsers import DocumentParsingError, parse_document
from embeddings.encoder import EmbeddingEncoder

logger = logging.getLogger(__name__)
router = Router()

ADMIN_ONLY_MESSAGE = "Эта команда доступна только администраторам бота."


def _is_admin(message: Message, settings: Settings) -> bool:
    """Проверяет, входит ли chat_id отправителя в список админов."""
    return message.chat.id in settings.admin_chat_ids


@router.message(Command("kb_add"))
async def cmd_kb_add(
    message: Message, state: FSMContext, settings: Settings
) -> None:
    if not _is_admin(message, settings):
        await message.answer(ADMIN_ONLY_MESSAGE)
        return

    await state.set_state(AddKnowledgeDocumentStates.waiting_for_file)
    await message.answer(
        "Пришлите документ для базы знаний — поддерживаются форматы "
        "PDF, DOCX, XLSX, TXT. Или /cancel для отмены."
    )


@router.message(
    Command("cancel"), StateFilter(AddKnowledgeDocumentStates.waiting_for_file)
)
async def cmd_cancel_kb_add(message: Message, state: FSMContext) -> None:
    await state.clear()
    await message.answer("Добавление документа отменено.")


@router.message(StateFilter(AddKnowledgeDocumentStates.waiting_for_file), F.document)
async def kb_add_receive_file(
    message: Message,
    bot: Bot,
    state: FSMContext,
    repository: KnowledgeRepository,
    encoder: EmbeddingEncoder,
    settings: Settings,
) -> None:
    document = message.document
    if document is None:
        # Гарантировано фильтром F.document, но mypy/чтение кода
        # требуют явной проверки на None.
        return

    status_message = await message.answer("Обрабатываю документ…")

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = Path(tmp_dir) / (document.file_name or "document")
        file_info = await bot.get_file(document.file_id)
        assert file_info.file_path is not None
        await bot.download_file(
            file_info.file_path, destination=str(local_path)
        )

        try:
            text, original_format = parse_document(str(local_path))
        except DocumentParsingError as exc:
            await status_message.edit_text(
                f"⚠️ Не удалось обработать файл: {exc}"
            )
            return

        chunks_text = split_text_into_chunks(
            text,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    if not chunks_text:
        await status_message.edit_text(
            "⚠️ В документе не нашлось текста для добавления в базу "
            "знаний."
        )
        return

    embeddings = encoder.encode_batch(chunks_text)

    knowledge_document = repository.add_document(
        admin_chat_id=message.chat.id,
        filename=document.file_name or "document",
        original_format=original_format,
    )
    repository.add_chunks(
        knowledge_document.id, list(zip(chunks_text, embeddings))
    )

    await state.clear()
    await status_message.edit_text(
        f"✅ Документ «{document.file_name}» добавлен в базу знаний "
        f"(id={knowledge_document.id}), фрагментов: {len(chunks_text)}."
    )


@router.message(StateFilter(AddKnowledgeDocumentStates.waiting_for_file))
async def kb_add_wrong_content(message: Message) -> None:
    await message.answer(
        "Пришлите файл документом (не как текст и не как фото) — "
        "PDF, DOCX, XLSX или TXT. Либо /cancel для отмены."
    )


@router.message(Command("kb_list"))
async def cmd_kb_list(
    message: Message, settings: Settings, repository: KnowledgeRepository
) -> None:
    if not _is_admin(message, settings):
        await message.answer(ADMIN_ONLY_MESSAGE)
        return

    documents = repository.list_documents()
    if not documents:
        await message.answer(
            "База знаний пуста. Добавьте документ командой /kb_add"
        )
        return

    lines = ["📚 Документы базы знаний:\n"]
    for doc in documents:
        lines.append(
            f"#{doc.id} {doc.filename} ({doc.original_format}, "
            f"фрагментов: {len(doc.chunks)}) — "
            f"{doc.created_at:%Y-%m-%d %H:%M}"
        )
    await message.answer("\n".join(lines))


@router.message(Command("kb_remove"))
async def cmd_kb_remove(
    message: Message, settings: Settings, repository: KnowledgeRepository
) -> None:
    if not _is_admin(message, settings):
        await message.answer(ADMIN_ONLY_MESSAGE)
        return

    args = (message.text or "").split(maxsplit=1)
    if len(args) < 2 or not args[1].strip().isdigit():
        await message.answer(
            "Использование: /kb_remove ID_документа, например: "
            "/kb_remove 3\nПосмотреть ID можно командой /kb_list"
        )
        return

    document_id = int(args[1].strip())
    removed = repository.remove_document(document_id)
    if removed:
        await message.answer(f"Документ #{document_id} удалён из базы знаний.")
    else:
        await message.answer(f"Документ #{document_id} не найден.")


@router.message(Command("stats"))
async def cmd_stats(
    message: Message, settings: Settings, repository: KnowledgeRepository
) -> None:
    if not _is_admin(message, settings):
        await message.answer(ADMIN_ONLY_MESSAGE)
        return

    stats = repository.get_stats()
    lines = [
        "📊 Статистика обращений клиентов:\n",
        f"Всего вопросов: {stats.total_questions}",
        f"Эскалировано человеку: {stats.escalated_count}",
    ]
    if stats.recent_escalated:
        lines.append("\nПоследние эскалированные вопросы:")
        for log in stats.recent_escalated:
            lines.append(
                f"— {log.question} ({log.created_at:%Y-%m-%d %H:%M})"
            )
    await message.answer("\n".join(lines))

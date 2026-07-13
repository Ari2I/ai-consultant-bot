"""
Обработчики для клиентов: обычные текстовые вопросы обрабатываются
через RAG-пайплайн, при низкой уверенности — эскалируются админам
вместе с информацией о клиенте.
"""

from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from bot.rate_limiter import RateLimiter
from config import Settings
from database.repository import KnowledgeRepository
from rag.pipeline import RagPipeline

logger = logging.getLogger(__name__)
router = Router()

HELP_TEXT = (
    "🤖 Я — ИИ-консультант, отвечаю на вопросы на основе базы знаний "
    "компании.\n\n"
    "Просто напишите ваш вопрос — я постараюсь ответить сразу. Если "
    "не найду точного ответа в базе знаний, передам вопрос менеджеру, "
    "и он свяжется с вами."
)


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    await message.answer("Здравствуйте!\n\n" + HELP_TEXT)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(HELP_TEXT)


def _format_escalation_notice(message: Message, question: str) -> str:
    """Формирует уведомление для админов об эскалированном вопросе."""
    user = message.from_user
    username = f"@{user.username}" if user and user.username else "без username"
    full_name = user.full_name if user else "неизвестно"
    return (
        "🔔 Клиент задал вопрос, на который бот не смог уверенно "
        "ответить:\n\n"
        f"От: {full_name} ({username}, chat_id={message.chat.id})\n"
        f"Вопрос: {question}"
    )


@router.message(F.text, ~F.text.startswith("/"))
async def handle_client_question(
    message: Message,
    bot: Bot,
    settings: Settings,
    repository: KnowledgeRepository,
    rag_pipeline: RagPipeline,
    rate_limiter: RateLimiter,
) -> None:
    question = (message.text or "").strip()
    if not question:
        return

    if message.chat.id in settings.admin_chat_ids:
        # Админ написал обычным текстом, а не командой — не гоняем
        # это через RAG (не тратим лимит/токены на служебные
        # сообщения), а просто подсказываем доступные команды.
        await message.answer(
            "Вы администратор этого бота. Доступные команды: "
            "/kb_add, /kb_list, /kb_remove, /stats"
        )
        return

    if not rate_limiter.allow(message.chat.id):
        await message.answer(
            "Слишком много вопросов подряд — подождите немного и "
            "попробуйте снова."
        )
        return

    result = await asyncio.to_thread(rag_pipeline.answer_question, question)

    repository.log_conversation(
        customer_chat_id=message.chat.id,
        question=question,
        answer=result.answer,
        similarity_score=result.similarity_score,
        escalated=result.escalated,
        matched_chunk_ids=result.matched_chunk_ids or None,
    )

    if result.escalated:
        await message.answer(
            "Не могу уверенно ответить на этот вопрос — передал его "
            "менеджеру, он свяжется с вами."
        )
        notice = _format_escalation_notice(message, question)
        for admin_chat_id in settings.admin_chat_ids:
            try:
                await bot.send_message(admin_chat_id, notice)
            except Exception:  # noqa: BLE001 — сбой уведомления одного админа
                logger.exception(
                    "Не удалось переслать эскалацию админу %s",
                    admin_chat_id,
                )
        return

    await message.answer(result.answer or "")


@router.message(F.text.startswith("/"))
async def unknown_command(message: Message) -> None:
    await message.answer("Неизвестная команда. Используйте /help для справки.")

"""
Точка входа в приложение "ИИ-консультант для малого бизнеса".

Запуск:
    python main.py
"""

from __future__ import annotations

import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiogram.client.session.aiohttp import AiohttpSession

from bot.admin_handlers import router as admin_router
from bot.client_handlers import router as client_router
from bot.rate_limiter import RateLimiter
from config import load_settings
from database.repository import KnowledgeRepository
from embeddings.encoder import EmbeddingEncoder
from llm.deepseek_client import DeepSeekClient
from rag.pipeline import RagPipeline


def configure_logging() -> None:
    """Настраивает базовое логирование приложения."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


async def main() -> None:
    configure_logging()
    settings = load_settings()

    # Если Telegram Bot API заблокирован на уровне сети (актуально
    # для многих сетей в России), задайте PROXY_URL в .env — сессия
    # бота пойдёт через него. Если переменная пустая — прямое
    # подключение, как обычно.
    session = (
        AiohttpSession(proxy=settings.bot_proxy_url)
        if settings.bot_proxy_url
        else None
    )
    bot = Bot(token=settings.bot_token, session=session)
    dispatcher = Dispatcher()
    # Порядок важен: сначала админские обработчики (проверяют
    # ADMIN_CHAT_IDS и явно отвечают отказом не-админам на команды
    # вида /kb_add), затем клиентский Q&A-флоу с обработчиком
    # "неизвестная команда" в конце.
    dispatcher.include_router(admin_router)
    dispatcher.include_router(client_router)

    repository = KnowledgeRepository(settings.database_url)
    # Загрузка модели эмбеддингов может занять время при первом
    # запуске (скачивание файлов модели из HuggingFace Hub) — см.
    # DEPLOYMENT.md про необходимость доступа в интернет на сервере
    # при первом старте.
    encoder = EmbeddingEncoder(settings.embedding_model_name)
    llm_client = DeepSeekClient(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        max_retries=settings.deepseek_max_retries,
        retry_backoff_seconds=settings.deepseek_retry_backoff_seconds,
        request_timeout=settings.deepseek_request_timeout,
    )
    rag_pipeline = RagPipeline(
        repository=repository,
        encoder=encoder,
        llm_client=llm_client,
        similarity_threshold=settings.similarity_threshold,
        top_k_chunks=settings.top_k_chunks,
    )
    rate_limiter = RateLimiter(settings.rate_limit_per_minute)

    try:
        await dispatcher.start_polling(
            bot,
            settings=settings,
            repository=repository,
            encoder=encoder,
            rag_pipeline=rag_pipeline,
            rate_limiter=rate_limiter,
        )
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())

"""
Конфигурация приложения "ИИ-консультант для малого бизнеса".

Все параметры читаются из переменных окружения (файл .env в корне
проекта). Пример заполнения — см. .env.example.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()


def _get_int_list(name: str) -> List[int]:
    """Разбирает переменную окружения вида '123,456' в список int."""
    raw = os.getenv(name, "")
    result: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            result.append(int(part))
        except ValueError:
            continue
    return result


@dataclass(frozen=True)
class Settings:
    """Настройки приложения."""

    bot_token: str
    admin_chat_ids: List[int]
    database_url: str
    bot_proxy_url: Optional[str]

    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_model: str
    deepseek_max_retries: int
    deepseek_retry_backoff_seconds: float
    deepseek_request_timeout: int

    embedding_model_name: str

    similarity_threshold: float
    top_k_chunks: int
    chunk_size: int
    chunk_overlap: int

    rate_limit_per_minute: int


def load_settings() -> Settings:
    """Загружает и валидирует настройки из переменных окружения."""
    bot_token = os.getenv("BOT_TOKEN", "").strip()
    if not bot_token:
        raise RuntimeError(
            "Не задан BOT_TOKEN. Укажите токен Telegram-бота "
            "в файле .env (см. .env.example)"
        )

    admin_chat_ids = _get_int_list("ADMIN_CHAT_IDS")
    if not admin_chat_ids:
        raise RuntimeError(
            "Не задан ADMIN_CHAT_IDS. Укажите хотя бы один chat_id "
            "владельца бизнеса через запятую, например: "
            "ADMIN_CHAT_IDS=123456789"
        )

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not deepseek_api_key:
        raise RuntimeError(
            "Не задан DEEPSEEK_API_KEY. Получите ключ на "
            "https://platform.deepseek.com и укажите его в .env"
        )

    return Settings(
        bot_token=bot_token,
        admin_chat_ids=admin_chat_ids,
        database_url=os.getenv(
            "DATABASE_URL", "sqlite:///knowledge_base.db"
        ),
        bot_proxy_url=os.getenv("PROXY_URL", "").strip() or None,
        deepseek_api_key=deepseek_api_key,
        deepseek_base_url=os.getenv(
            "DEEPSEEK_BASE_URL", "https://api.deepseek.com"
        ),
        # deepseek-v4-flash — актуальная (на 07.2026) быстрая и дешёвая
        # модель, достаточная для ответов на основе готового RAG-
        # контекста. Устаревшие имена deepseek-chat / deepseek-reasoner
        # прекращают поддержку 24.07.2026, поэтому используется новый
        # идентификатор, а не легаси-алиас.
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        deepseek_max_retries=int(os.getenv("DEEPSEEK_MAX_RETRIES", "2")),
        deepseek_retry_backoff_seconds=float(
            os.getenv("DEEPSEEK_RETRY_BACKOFF_SECONDS", "1.0")
        ),
        deepseek_request_timeout=int(
            os.getenv("DEEPSEEK_REQUEST_TIMEOUT", "30")
        ),
        # Многоязычная модель эмбеддингов (поддерживает русский),
        # работает локально на CPU через sentence-transformers, без
        # платных API-вызовов.
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME",
            "paraphrase-multilingual-MiniLM-L12-v2",
        ),
        similarity_threshold=float(
            os.getenv("SIMILARITY_THRESHOLD", "0.75")
        ),
        top_k_chunks=int(os.getenv("TOP_K_CHUNKS", "5")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "50")),
        rate_limit_per_minute=int(
            os.getenv("RATE_LIMIT_PER_MINUTE", "10")
        ),
    )

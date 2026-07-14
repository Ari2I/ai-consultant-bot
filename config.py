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


def _get_proxy_url() -> Optional[str]:
    """
    Читает и валидирует PROXY_URL из окружения.

    Явно проверяем наличие схемы (socks5://, socks4://, http://,
    https://) на этапе загрузки настроек — без этой проверки
    некорректный PROXY_URL (например, просто "1.2.3.4:1080" без
    схемы) падает глубоко внутри aiogram/python-socks с
    труднопонятным "Invalid scheme component: ", а не с понятным
    сообщением о том, что не так в .env.
    """
    raw = os.getenv("PROXY_URL", "").strip()
    if not raw:
        return None

    valid_schemes = ("socks5://", "socks4://", "http://", "https://")
    if not raw.lower().startswith(valid_schemes):
        raise RuntimeError(
            f"Некорректный PROXY_URL: '{raw}'. Не указана схема "
            f"прокси. Ожидается один из форматов: "
            f"socks5://user:password@host:port, "
            f"socks5://host:port, http://host:port и т.д."
        )
    return raw


def _get_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "да"}


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

    gigachat_credentials: str
    gigachat_scope: str
    gigachat_model: str
    gigachat_verify_ssl_certs: bool
    gigachat_ca_bundle_file: Optional[str]
    gigachat_max_retries: int
    gigachat_retry_backoff_seconds: float
    gigachat_request_timeout: int

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
            "в файле .env"
        )

    admin_chat_ids = _get_int_list("ADMIN_CHAT_IDS")
    if not admin_chat_ids:
        raise RuntimeError(
            "Не задан ADMIN_CHAT_IDS. Укажите хотя бы один chat_id "
            "владельца бизнеса через запятую, например: "
            "ADMIN_CHAT_IDS=123456789"
        )

    gigachat_credentials = os.getenv("GIGACHAT_CREDENTIALS", "").strip()
    if not gigachat_credentials:
        raise RuntimeError(
            "Не задан GIGACHAT_CREDENTIALS. Получите Authorization key "
            "в личном кабинете https://developers.sber.ru/studio/ "
            "(раздел проекта GigaChat API → Настройки API → Получить "
            "ключ) и укажите его в .env"
        )

    return Settings(
        bot_token=bot_token,
        admin_chat_ids=admin_chat_ids,
        database_url=os.getenv(
            "DATABASE_URL", "sqlite:///knowledge_base.db"
        ),
        bot_proxy_url=_get_proxy_url(),
        gigachat_credentials=gigachat_credentials,
        # GIGACHAT_API_PERS — доступ для физических лиц (подходит для
        # разработки и небольших проектов). Для юрлиц/ИП — B2B (пакеты)
        # или CORP (pay-as-you-go) — см. .env.example.
        gigachat_scope=os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS"),
        # GigaChat-2 — актуальная (на 07.2026) базовая модель. Модели
        # первого поколения (GigaChat, GigaChat-Pro, GigaChat-Max)
        # больше не существуют как отдельные модели — Sber сам
        # перенаправляет такие запросы на GigaChat-2/-Pro/-Max, но
        # правильнее сразу использовать актуальное имя.
        gigachat_model=os.getenv("GIGACHAT_MODEL", "GigaChat-2"),
        # Сертификат НУЦ Минцифры обязателен для TLS-соединения с
        # GigaChat API. По умолчанию проверка включена (безопасно) —
        # выключать (GIGACHAT_VERIFY_SSL_CERTS=false) стоит только
        # для локальной разработки, если сертификат ещё не установлен
        # (см. README/DEPLOYMENT.md).
        gigachat_verify_ssl_certs=_get_bool(
            "GIGACHAT_VERIFY_SSL_CERTS", True
        ),
        gigachat_ca_bundle_file=(
            os.getenv("GIGACHAT_CA_BUNDLE_FILE", "").strip() or None
        ),
        gigachat_max_retries=int(os.getenv("GIGACHAT_MAX_RETRIES", "2")),
        gigachat_retry_backoff_seconds=float(
            os.getenv("GIGACHAT_RETRY_BACKOFF_SECONDS", "1.0")
        ),
        gigachat_request_timeout=int(
            os.getenv("GIGACHAT_REQUEST_TIMEOUT", "30")
        ),
        # Многоязычная модель эмбеддингов (поддерживает русский),
        # работает локально на CPU через sentence-transformers, без
        # платных API-вызовов. Не связана с GigaChat — эмбеддинги
        # остаются локальными независимо от провайдера генерации.
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

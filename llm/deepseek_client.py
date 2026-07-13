"""
Клиент для генерации ответов через DeepSeek API (OpenAI-совместимый).

Эндпоинт и SDK сверены как актуальные на 07.2026: DeepSeek API
доступен по адресу https://api.deepseek.com/chat/completions через
официальный пакет `openai` с изменённым base_url — отдельного SDK у
DeepSeek нет и не требуется. Модель по умолчанию — deepseek-v4-flash
(см. config.py); устаревшие имена deepseek-chat / deepseek-reasoner
прекращают поддержку 24.07.2026, поэтому используется актуальный
идентификатор.

Устойчивость к временным сбоям реализована по тому же принципу, что
и в предыдущих проектах: повтор с задержкой (backoff) при временных
ошибках (таймаут, обрыв соединения, HTTP 429/5xx), без повтора при
окончательных ошибках (401 — неверный ключ, 403 — доступ запрещён,
400 — некорректный запрос).
"""

from __future__ import annotations

import logging
import time
from typing import Any, List, Optional

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

# Инструкция модели: отвечать строго по контексту и не придумывать
# факты — прямое требование бизнес-логики проекта (см. план: бот не
# должен выдумывать ответы, а эскалировать вопрос человеку).
SYSTEM_PROMPT = (
    "Ты — вежливый ИИ-консультант небольшого бизнеса в Telegram. "
    "Отвечай ТОЛЬКО на основе предоставленного ниже контекста из "
    "базы знаний компании. Если в контексте нет ответа на вопрос — "
    "прямо скажи, что не располагаешь этой информацией, и не "
    "придумывай факты. Отвечай кратко, по делу и на русском языке."
)

# Ошибки, при которых имеет смысл повторить запрос — временные
# проблемы сети или перегрузка сервера, которые обычно проходят сами.
_RETRYABLE_ERRORS = (
    APIConnectionError,
    APITimeoutError,
    RateLimitError,
    InternalServerError,
)

# Окончательные ошибки — повторять запрос бессмысленно и вредно
# (лишний расход лимита при 429 здесь не аргумент, так как это не
# 429; неверный ключ или запрещённый доступ не исправятся повтором).
_NON_RETRYABLE_ERRORS = (
    AuthenticationError,
    PermissionDeniedError,
    BadRequestError,
)


class DeepSeekApiError(Exception):
    """Окончательная ошибка при обращении к DeepSeek API."""


class DeepSeekClient:
    """Обёртка над DeepSeek chat completions с ретраями."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        request_timeout: int = 30,
        client: Optional[Any] = None,
    ) -> None:
        """
        Аргументы:
            client: готовый клиент (для тестов/внедрения зависимости).
                Если не передан, создаётся реальный openai.OpenAI с
                указанными api_key/base_url/timeout.
        """
        self._client = client or OpenAI(
            api_key=api_key, base_url=base_url, timeout=request_timeout
        )
        self.model = model
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds

    def generate_answer(self, question: str, context: str) -> str:
        """
        Генерирует ответ на вопрос клиента на основе контекста RAG.

        Аргументы:
            question: вопрос клиента.
            context: релевантные фрагменты базы знаний, найденные
                поиском (см. rag/search.py).

        Возвращает:
            Текст ответа модели.

        Исключения:
            DeepSeekApiError: при окончательной ошибке API — либо
                неповторяемая ошибка (401/403/400), либо повторяемая
                ошибка после исчерпания всех попыток.
        """
        messages: List[Any] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Контекст из базы знаний компании:\n{context}\n\n"
                    f"Вопрос клиента: {question}"
                ),
            },
        ]

        attempt = 0
        while True:
            try:
                response = self._client.chat.completions.create(
                    model=self.model, messages=messages
                )
                return response.choices[0].message.content or ""
            except _NON_RETRYABLE_ERRORS as exc:
                raise DeepSeekApiError(
                    f"Ошибка запроса к DeepSeek API (без повтора): {exc}"
                ) from exc
            except _RETRYABLE_ERRORS as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise DeepSeekApiError(
                        f"DeepSeek API недоступен после "
                        f"{self.max_retries} повторов: {exc}"
                    ) from exc
                wait_seconds = self.retry_backoff_seconds * attempt
                logger.info(
                    "Временная ошибка DeepSeek API "
                    "(попытка %d из %d): %s. Повтор через %.1f с.",
                    attempt,
                    self.max_retries,
                    exc,
                    wait_seconds,
                )
                time.sleep(wait_seconds)

"""
Клиент для генерации ответов через GigaChat API (Сбер).

Используется официальный Python SDK `gigachat` (проект ai-forever) —
сверено по факту установленной версии на 07.2026, а не по памяти:
- Авторизация — OAuth2 по Authorization key (GIGACHAT_CREDENTIALS,
  строка client_id:client_secret в base64) + scope
  (GIGACHAT_API_PERS/B2B/CORP). SDK сам получает и обновляет access-
  токен (живёт около 30 минут) — вручную это делать не нужно.
- Для TLS-соединения обязателен корневой сертификат НУЦ Минцифры —
  без него запросы падают с ошибкой проверки сертификата. Подробности
  — в README/DEPLOYMENT.md.
- Модель по умолчанию — GigaChat-2 (актуальное на 07.2026 поколение;
  модели первого поколения GigaChat/GigaChat-Pro/GigaChat-Max больше
  не существуют отдельно и автоматически перенаправляются на
  GigaChat-2/-Pro/-Max на стороне Sber).
- Используется "корневой" интерфейс client.chat(payload) с
  gigachat.models.Chat/Messages — это стабильный, документированный
  способ вызова (в отличие от совсем нового client.chat.create(...),
  который на момент написания ещё не имеет устоявшейся публичной
  сигнатуры для мульти-сообщений с системным промптом).

Устойчивость к временным сбоям реализована по тому же принципу, что
и в остальных внешних интеграциях проекта: повтор с задержкой
(backoff) при временных ошибках (лимит запросов, 5xx), без повтора
при окончательных ошибках (401/403/400/404/413/422).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

from gigachat import GigaChat
from gigachat.exceptions import (
    AuthenticationError,
    BadRequestError,
    ForbiddenError,
    NotFoundError,
    RateLimitError,
    RequestEntityTooLargeError,
    ServerError,
    UnprocessableEntityError,
)
from gigachat.models import Chat, Messages, MessagesRole

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
# проблемы (лимит запросов, перегрузка сервера), которые обычно
# проходят сами.
_RETRYABLE_ERRORS = (RateLimitError, ServerError)

# Окончательные ошибки — повторять запрос бессмысленно: неверные
# credentials/scope, запрещённый доступ, некорректный запрос,
# несуществующий эндпоинт/модель, слишком большой запрос — ни одно
# из этого не исправится повтором той же попытки.
_NON_RETRYABLE_ERRORS = (
    AuthenticationError,
    ForbiddenError,
    BadRequestError,
    NotFoundError,
    RequestEntityTooLargeError,
    UnprocessableEntityError,
)


class GigaChatApiError(Exception):
    """Окончательная ошибка при обращении к GigaChat API."""


class GigaChatClient:
    """Обёртка над GigaChat chat completions с ретраями."""

    def __init__(
        self,
        credentials: str,
        scope: str,
        model: str,
        verify_ssl_certs: bool = True,
        ca_bundle_file: Optional[str] = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 1.0,
        request_timeout: int = 30,
        client: Optional[Any] = None,
    ) -> None:
        """
        Аргументы:
            client: готовый клиент (для тестов/внедрения зависимости).
                Если не передан, создаётся реальный gigachat.GigaChat
                с указанными параметрами. SDK сам получает и обновляет
                OAuth-токен по credentials/scope — access_token
                передавать не нужно.
        """
        self._client = client or GigaChat(
            credentials=credentials,
            scope=scope,
            model=model,
            verify_ssl_certs=verify_ssl_certs,
            ca_bundle_file=ca_bundle_file,
            timeout=request_timeout,
            # Ретраи делаем сами (см. generate_answer) — единообразно
            # с остальными внешними интеграциями проекта, поэтому
            # встроенный механизм повторов SDK отключаем явно.
            max_retries=0,
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
            GigaChatApiError: при окончательной ошибке API — либо
                неповторяемая ошибка (401/403/400/404/413/422), либо
                повторяемая ошибка после исчерпания всех попыток.
        """
        payload = Chat(
            model=self.model,
            messages=[
                Messages(role=MessagesRole.SYSTEM, content=SYSTEM_PROMPT),
                Messages(
                    role=MessagesRole.USER,
                    content=(
                        f"Контекст из базы знаний компании:\n{context}\n\n"
                        f"Вопрос клиента: {question}"
                    ),
                ),
            ],
        )

        attempt = 0
        while True:
            try:
                response = self._client.chat(payload)
                return response.choices[0].message.content or ""
            except _NON_RETRYABLE_ERRORS as exc:
                raise GigaChatApiError(
                    f"Ошибка запроса к GigaChat API (без повтора): {exc}"
                ) from exc
            except _RETRYABLE_ERRORS as exc:
                attempt += 1
                if attempt > self.max_retries:
                    raise GigaChatApiError(
                        f"GigaChat API недоступен после "
                        f"{self.max_retries} повторов: {exc}"
                    ) from exc
                wait_seconds = self.retry_backoff_seconds * attempt
                logger.info(
                    "Временная ошибка GigaChat API "
                    "(попытка %d из %d): %s. Повтор через %.1f с.",
                    attempt,
                    self.max_retries,
                    exc,
                    wait_seconds,
                )
                time.sleep(wait_seconds)

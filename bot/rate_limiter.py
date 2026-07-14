"""
Простой rate limiter в памяти для защиты от флуда вопросами.

Ограничивает число вопросов одного клиента в минуту — без этого
случайный флуд (или намеренная атака) мог бы бесконтрольно тратить
платные вызовы GigaChat API. Хранит только временные метки последних
сообщений каждого chat_id, без персистентности: при перезапуске бота
лимиты обнуляются — это приемлемо для защиты от флуда (не финансовый
биллинг, где точность имела бы значение).
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict


class RateLimiter:
    """Ограничивает количество обращений одного chat_id в минуту."""

    def __init__(self, max_per_minute: int) -> None:
        self.max_per_minute = max_per_minute
        self._timestamps: Dict[int, Deque[float]] = defaultdict(deque)

    def allow(self, chat_id: int) -> bool:
        """
        Проверяет, можно ли обработать ещё одно обращение chat_id.

        Возвращает:
            True и регистрирует обращение, если лимit не превышен.
            False, если за последние 60 секунд обращений уже
            максимально допустимое количество.
        """
        now = time.monotonic()
        window_start = now - 60.0

        timestamps = self._timestamps[chat_id]
        while timestamps and timestamps[0] < window_start:
            timestamps.popleft()

        if len(timestamps) >= self.max_per_minute:
            return False

        timestamps.append(now)
        return True

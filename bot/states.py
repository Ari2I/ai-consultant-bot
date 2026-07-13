"""FSM-состояния диалогов бота."""

from __future__ import annotations

from aiogram.fsm.state import State, StatesGroup


class AddKnowledgeDocumentStates(StatesGroup):
    """
    Диалог добавления документа в базу знаний (/kb_add).

    Единственный шаг: ожидание файла. Парсинг, чанкинг и вычисление
    эмбеддингов выполняются сразу после получения файла, без
    дополнительных вопросов пользователю — формат определяется
    автоматически по расширению.
    """

    waiting_for_file = State()

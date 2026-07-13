"""
Обёртка над локальной моделью эмбеддингов (sentence-transformers).

Модель работает полностью локально на CPU, без внешних API-вызовов и
без затрат за токены — единственный платный вызов в проекте это
генерация ответа через DeepSeek (см. llm/deepseek_client.py). Сам файл
модели скачивается один раз при первом использовании из HuggingFace
Hub и затем кэшируется локально (см. README/DEPLOYMENT.md — на
сервере для первого запуска нужен доступ в интернет к huggingface.co).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence


class EmbeddingEncoder:
    """
    Вычисляет векторные представления (эмбеддинги) текста.

    Параметр `model` предназначен для тестирования и внедрения
    зависимостей (dependency injection): если он передан, реальная
    модель sentence-transformers не загружается вообще — это
    позволяет тестировать остальной код проекта (RAG-пайплайн,
    обработчики бота) без необходимости скачивать и держать в памяти
    тяжёлую ML-модель в CI/тестовом окружении.
    """

    def __init__(self, model_name: str, model: Optional[Any] = None) -> None:
        self.model_name = model_name
        if model is not None:
            self._model = model
        else:
            # Импорт внутри конструктора, а не на уровне модуля — по
            # той же причине, что и лениво импортируемый Playwright в
            # прошлом проекте: остальной код может импортироваться и
            # тестироваться даже в окружении, где sentence-transformers
            # ещё не установлен или устанавливается отдельно.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)

    def encode(self, text: str) -> List[float]:
        """Возвращает эмбеддинг одного текста в виде списка float."""
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: Sequence[str]) -> List[List[float]]:
        """
        Возвращает эмбеддинги для списка текстов.

        Пакетное вычисление (а не по одному тексту в цикле) —
        существенно быстрее для документов с десятками фрагментов.
        """
        vectors = self._model.encode(list(texts), convert_to_numpy=True)
        return [vector.tolist() for vector in vectors]

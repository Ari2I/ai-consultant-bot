"""
Тесты для rag.pipeline.RagPipeline.

Используется реальный KnowledgeRepository (SQLite in-memory) для
реалистичности, но EmbeddingEncoder и DeepSeekClient полностью
заменены фиктивными зависимостями через параметры конструктора —
без сети, без реальной ML-модели и без токенов DeepSeek.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np

from database.repository import KnowledgeRepository
from embeddings.encoder import EmbeddingEncoder
from llm.deepseek_client import DeepSeekApiError, DeepSeekClient
from rag.pipeline import RagPipeline


def _make_success_llm_response(text: str):
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


class RagPipelineTest(unittest.TestCase):
    def setUp(self):
        self.repository = KnowledgeRepository("sqlite:///:memory:")
        document = self.repository.add_document(1, "faq.docx", "docx")
        # Эмбеддинги подобраны так, чтобы вопрос про режим работы был
        # явно ближе к первому чанку, чем ко второму.
        self.repository.add_chunks(
            document.id,
            [
                ("Магазин работает с 10:00 до 20:00.", [1.0, 0.0, 0.0]),
                ("Доставка по Москве 300 рублей.", [0.0, 1.0, 0.0]),
            ],
        )

        self.fake_llm_client_backend = MagicMock()
        self.fake_model = MagicMock()

        self.encoder = EmbeddingEncoder("fake-model", model=self.fake_model)
        self.llm_client = DeepSeekClient(
            api_key="k",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            client=self.fake_llm_client_backend,
        )

    def _make_pipeline(self, threshold: float = 0.75, top_k: int = 5):
        return RagPipeline(
            repository=self.repository,
            encoder=self.encoder,
            llm_client=self.llm_client,
            similarity_threshold=threshold,
            top_k_chunks=top_k,
        )

    def test_answers_when_similarity_above_threshold(self):
        self.fake_model.encode.return_value = np.array([[1.0, 0.0, 0.0]])
        self.fake_llm_client_backend.chat.completions.create.return_value = (
            _make_success_llm_response("Работаем с 10:00 до 20:00.")
        )
        pipeline = self._make_pipeline(threshold=0.75)

        result = pipeline.answer_question("Когда вы работаете?")

        self.assertFalse(result.escalated)
        self.assertEqual(result.answer, "Работаем с 10:00 до 20:00.")
        self.assertAlmostEqual(result.similarity_score, 1.0, places=5)
        self.assertEqual(len(result.matched_chunk_ids), 2)

    def test_escalates_when_similarity_below_threshold(self):
        # Вектор, ортогональный обоим чанкам, — низкое сходство.
        self.fake_model.encode.return_value = np.array([[0.0, 0.0, 1.0]])
        pipeline = self._make_pipeline(threshold=0.75)

        result = pipeline.answer_question("Делаете ли вы татуировки?")

        self.assertTrue(result.escalated)
        self.assertIsNone(result.answer)
        self.fake_llm_client_backend.chat.completions.create.assert_not_called()

    def test_escalates_when_knowledge_base_is_empty(self):
        empty_repository = KnowledgeRepository("sqlite:///:memory:")
        pipeline = RagPipeline(
            repository=empty_repository,
            encoder=self.encoder,
            llm_client=self.llm_client,
            similarity_threshold=0.75,
            top_k_chunks=5,
        )

        result = pipeline.answer_question("Любой вопрос")

        self.assertTrue(result.escalated)
        self.assertIsNone(result.similarity_score)
        self.fake_model.encode.assert_not_called()

    def test_escalates_when_llm_fails_after_retries(self):
        self.fake_model.encode.return_value = np.array([[1.0, 0.0, 0.0]])
        self.fake_llm_client_backend.chat.completions.create.side_effect = (
            RuntimeError("недостижимо")
        )
        # Подменяем generate_answer напрямую, чтобы гарантированно
        # получить DeepSeekApiError без завязки на конкретный тип
        # исключения openai.
        self.llm_client.generate_answer = MagicMock(
            side_effect=DeepSeekApiError("DeepSeek недоступен")
        )
        pipeline = self._make_pipeline(threshold=0.75)

        result = pipeline.answer_question("Когда вы работаете?")

        self.assertTrue(result.escalated)
        self.assertIsNone(result.answer)
        self.assertIsNotNone(result.error)
        self.assertEqual(len(result.matched_chunk_ids), 2)


if __name__ == "__main__":
    unittest.main()

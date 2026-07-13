"""
Тесты для embeddings.encoder.EmbeddingEncoder.

Реальная модель sentence-transformers не используется и не
скачивается — вместо неё передаётся фиктивная модель (см. параметр
`model` конструктора), возвращающая заранее заданные numpy-векторы.
Это делает тесты быстрыми и не зависящими от загрузки ML-модели.
"""

import unittest
from unittest.mock import MagicMock

import numpy as np

from embeddings.encoder import EmbeddingEncoder


class EmbeddingEncoderTest(unittest.TestCase):
    def test_encode_batch_converts_numpy_to_list_of_lists(self):
        fake_model = MagicMock()
        fake_model.encode.return_value = np.array(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )
        encoder = EmbeddingEncoder("fake-model-name", model=fake_model)

        result = encoder.encode_batch(["текст 1", "текст 2"])

        self.assertEqual(result, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        fake_model.encode.assert_called_once_with(
            ["текст 1", "текст 2"], convert_to_numpy=True
        )

    def test_encode_single_text_returns_single_vector(self):
        fake_model = MagicMock()
        fake_model.encode.return_value = np.array([[0.7, 0.8]])
        encoder = EmbeddingEncoder("fake-model-name", model=fake_model)

        result = encoder.encode("один текст")

        self.assertEqual(result, [0.7, 0.8])

    def test_stores_model_name(self):
        fake_model = MagicMock()
        encoder = EmbeddingEncoder(
            "paraphrase-multilingual-MiniLM-L12-v2", model=fake_model
        )
        self.assertEqual(
            encoder.model_name, "paraphrase-multilingual-MiniLM-L12-v2"
        )


if __name__ == "__main__":
    unittest.main()

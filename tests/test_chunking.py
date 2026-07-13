"""Тесты для documents.chunking.split_text_into_chunks."""

import unittest

from documents.chunking import split_text_into_chunks


class SplitTextIntoChunksTest(unittest.TestCase):
    def test_empty_text_returns_empty_list(self):
        self.assertEqual(split_text_into_chunks(""), [])
        self.assertEqual(split_text_into_chunks("   \n\n   "), [])

    def test_short_text_returns_single_chunk(self):
        text = "Магазин работает с 10:00 до 20:00."
        chunks = split_text_into_chunks(text, chunk_size=500, chunk_overlap=50)
        self.assertEqual(chunks, [text])

    def test_invalid_overlap_raises(self):
        with self.assertRaises(ValueError):
            split_text_into_chunks("текст", chunk_size=100, chunk_overlap=100)
        with self.assertRaises(ValueError):
            split_text_into_chunks("текст", chunk_size=100, chunk_overlap=150)

    def test_splits_multiple_paragraphs_into_several_chunks(self):
        paragraphs = [f"Абзац номер {i}. " * 5 for i in range(10)]
        text = "\n\n".join(paragraphs)

        chunks = split_text_into_chunks(text, chunk_size=100, chunk_overlap=20)

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            # Небольшое допущение из-за перекрытия — не должно расти
            # бесконтрольно относительно лимита.
            self.assertLessEqual(len(chunk), 100 + 20 + 2)

    def test_overlap_carries_tail_of_previous_chunk(self):
        paragraphs = [f"Абзац {i} с некоторым содержимым текста." for i in range(8)]
        text = "\n\n".join(paragraphs)

        chunks = split_text_into_chunks(text, chunk_size=80, chunk_overlap=20)

        self.assertGreaterEqual(len(chunks), 2)
        # Конец первого чанка должен встречаться в начале второго —
        # это и есть перекрытие.
        tail_of_first = chunks[0][-20:]
        self.assertIn(tail_of_first, chunks[1])

    def test_long_paragraph_without_periods_is_hard_split(self):
        # Один абзац без знаков препинания длиннее chunk_size — должен
        # быть жёстко нарезан по символам, а не потерян.
        long_word_paragraph = "а" * 250
        chunks = split_text_into_chunks(
            long_word_paragraph, chunk_size=100, chunk_overlap=10
        )
        self.assertGreater(len(chunks), 1)
        joined = "".join(chunks).replace("\n\n", "")
        # Все исходные символы должны присутствовать (с учётом
        # возможных повторов из-за перекрытия).
        self.assertIn("а" * 90, joined)

    def test_paragraph_split_by_sentences_when_too_long(self):
        sentence = "Это одно короткое предложение."
        long_paragraph = " ".join([sentence] * 10)
        chunks = split_text_into_chunks(
            long_paragraph, chunk_size=120, chunk_overlap=20
        )
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(len(chunk) > 0)


if __name__ == "__main__":
    unittest.main()

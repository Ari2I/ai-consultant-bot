"""Тесты для rag.search.find_relevant_chunks."""

import unittest

from database.repository import ChunkRecord
from rag.search import find_relevant_chunks


def make_chunks():
    return [
        ChunkRecord(
            id=1, document_id=1, filename="faq.docx",
            chunk_text="Режим работы с 10 до 20",
            embedding=[1.0, 0.0, 0.0],
        ),
        ChunkRecord(
            id=2, document_id=1, filename="faq.docx",
            chunk_text="Доставка по Москве 300 рублей",
            embedding=[0.0, 1.0, 0.0],
        ),
        ChunkRecord(
            id=3, document_id=2, filename="price_list.xlsx",
            chunk_text="Футболка 1500 рублей",
            embedding=[0.0, 0.0, 1.0],
        ),
    ]


class FindRelevantChunksTest(unittest.TestCase):
    def test_returns_most_similar_chunk_first(self):
        chunks = make_chunks()
        results = find_relevant_chunks([1.0, 0.0, 0.0], chunks, top_k=3)

        self.assertEqual(results[0].chunk_id, 1)
        self.assertAlmostEqual(results[0].similarity, 1.0, places=5)

    def test_respects_top_k(self):
        chunks = make_chunks()
        results = find_relevant_chunks([1.0, 0.1, 0.0], chunks, top_k=2)
        self.assertEqual(len(results), 2)

    def test_empty_chunks_returns_empty_list(self):
        results = find_relevant_chunks([1.0, 0.0, 0.0], [], top_k=5)
        self.assertEqual(results, [])

    def test_zero_query_vector_returns_empty_list(self):
        chunks = make_chunks()
        results = find_relevant_chunks([0.0, 0.0, 0.0], chunks, top_k=3)
        self.assertEqual(results, [])

    def test_top_k_larger_than_chunk_count_returns_all(self):
        chunks = make_chunks()
        results = find_relevant_chunks([1.0, 0.0, 0.0], chunks, top_k=100)
        self.assertEqual(len(results), 3)

    def test_results_sorted_descending_by_similarity(self):
        chunks = make_chunks()
        results = find_relevant_chunks([0.5, 0.3, 0.1], chunks, top_k=3)
        similarities = [r.similarity for r in results]
        self.assertEqual(similarities, sorted(similarities, reverse=True))

    def test_orthogonal_vector_has_zero_similarity(self):
        chunks = [
            ChunkRecord(
                id=1, document_id=1, filename="a.txt",
                chunk_text="текст", embedding=[0.0, 1.0],
            )
        ]
        results = find_relevant_chunks([1.0, 0.0], chunks, top_k=1)
        self.assertAlmostEqual(results[0].similarity, 0.0, places=5)


if __name__ == "__main__":
    unittest.main()

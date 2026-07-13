"""Тесты для database.repository.KnowledgeRepository."""

import unittest

from database.repository import KnowledgeRepository


def make_sample_chunks():
    return [
        (
            "Магазин работает с 10:00 до 20:00 без выходных.",
            [0.1, 0.2, 0.3],
        ),
        (
            "Доставка по городу — 300 рублей, от 3000 рублей бесплатно.",
            [0.4, 0.5, 0.6],
        ),
    ]


class KnowledgeDocumentsTest(unittest.TestCase):
    def setUp(self):
        self.repository = KnowledgeRepository("sqlite:///:memory:")

    def test_add_and_list_documents(self):
        self.repository.add_document(1, "faq.docx", "docx")
        documents = self.repository.list_documents()
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].filename, "faq.docx")
        self.assertEqual(documents[0].original_format, "docx")

    def test_add_chunks_and_get_all_chunks_roundtrip(self):
        document = self.repository.add_document(1, "faq.docx", "docx")
        self.repository.add_chunks(document.id, make_sample_chunks())

        chunks = self.repository.get_all_chunks()

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].filename, "faq.docx")
        self.assertEqual(chunks[0].embedding, [0.1, 0.2, 0.3])
        self.assertEqual(chunks[1].embedding, [0.4, 0.5, 0.6])

    def test_remove_document_deletes_chunks_too(self):
        document = self.repository.add_document(1, "faq.docx", "docx")
        self.repository.add_chunks(document.id, make_sample_chunks())

        removed = self.repository.remove_document(document.id)

        self.assertTrue(removed)
        self.assertEqual(self.repository.list_documents(), [])
        self.assertEqual(self.repository.get_all_chunks(), [])

    def test_remove_document_returns_false_for_missing_id(self):
        removed = self.repository.remove_document(999)
        self.assertFalse(removed)

    def test_multiple_documents_are_independent(self):
        doc_a = self.repository.add_document(1, "faq.docx", "docx")
        doc_b = self.repository.add_document(1, "price_list.xlsx", "xlsx")
        self.repository.add_chunks(doc_a.id, make_sample_chunks())
        self.repository.add_chunks(
            doc_b.id, [("Цена футболки — 1500 руб.", [0.7, 0.8, 0.9])]
        )

        self.repository.remove_document(doc_a.id)
        chunks = self.repository.get_all_chunks()

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].filename, "price_list.xlsx")


class ConversationLogTest(unittest.TestCase):
    def setUp(self):
        self.repository = KnowledgeRepository("sqlite:///:memory:")

    def test_log_conversation_answered(self):
        log = self.repository.log_conversation(
            customer_chat_id=555,
            question="Какой у вас режим работы?",
            answer="С 10:00 до 20:00 без выходных.",
            similarity_score=0.91,
            escalated=False,
            matched_chunk_ids=[1, 2],
        )
        self.assertEqual(log.matched_chunk_ids, "1,2")
        self.assertFalse(log.escalated)

    def test_log_conversation_escalated_without_answer(self):
        log = self.repository.log_conversation(
            customer_chat_id=555,
            question="А делаете ли вы татуировки?",
            answer=None,
            similarity_score=0.2,
            escalated=True,
        )
        self.assertIsNone(log.answer)
        self.assertTrue(log.escalated)
        self.assertIsNone(log.matched_chunk_ids)

    def test_get_stats_counts_correctly(self):
        self.repository.log_conversation(
            1, "Вопрос 1", "Ответ 1", 0.9, escalated=False
        )
        self.repository.log_conversation(
            2, "Вопрос 2", None, 0.1, escalated=True
        )
        self.repository.log_conversation(
            3, "Вопрос 3", None, 0.15, escalated=True
        )

        stats = self.repository.get_stats()

        self.assertEqual(stats.total_questions, 3)
        self.assertEqual(stats.escalated_count, 2)
        self.assertEqual(len(stats.recent_escalated), 2)

    def test_get_stats_respects_recent_limit(self):
        for i in range(5):
            self.repository.log_conversation(
                i, f"Вопрос {i}", None, 0.1, escalated=True
            )

        stats = self.repository.get_stats(recent_limit=3)

        self.assertEqual(stats.escalated_count, 5)
        self.assertEqual(len(stats.recent_escalated), 3)


if __name__ == "__main__":
    unittest.main()

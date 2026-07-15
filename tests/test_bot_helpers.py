"""
Тесты для чистых вспомогательных функций обработчиков бота.

Полноценные объекты aiogram.types.Message не создаются — для этих
функций нужны только конкретные атрибуты (chat.id, from_user.*),
поэтому используется SimpleNamespace как лёгкая замена, без
необходимости поднимать Bot/Dispatcher.
"""

import unittest
from types import SimpleNamespace

from bot.admin_handlers import _is_admin
from bot.client_handlers import _format_escalation_notice, _is_plain_greeting
from config import Settings


def make_settings(admin_chat_ids):
    return Settings(
        bot_token="test-token",
        admin_chat_ids=admin_chat_ids,
        database_url="sqlite:///:memory:",
        bot_proxy_url=None,
        gigachat_credentials="test-credentials",
        gigachat_scope="GIGACHAT_API_PERS",
        gigachat_model="GigaChat-2",
        gigachat_verify_ssl_certs=True,
        gigachat_ca_bundle_file=None,
        gigachat_max_retries=2,
        gigachat_retry_backoff_seconds=1.0,
        gigachat_request_timeout=30,
        embedding_model_name="paraphrase-multilingual-MiniLM-L12-v2",
        similarity_threshold=0.75,
        top_k_chunks=5,
        chunk_size=500,
        chunk_overlap=50,
        rate_limit_per_minute=10,
    )


def make_message(chat_id, username=None, full_name="Иван Иванов"):
    from_user = SimpleNamespace(username=username, full_name=full_name)
    chat = SimpleNamespace(id=chat_id)
    return SimpleNamespace(chat=chat, from_user=from_user)


class IsAdminTest(unittest.TestCase):
    def test_admin_chat_id_returns_true(self):
        settings = make_settings(admin_chat_ids=[111, 222])
        message = make_message(chat_id=111)
        self.assertTrue(_is_admin(message, settings))

    def test_non_admin_chat_id_returns_false(self):
        settings = make_settings(admin_chat_ids=[111, 222])
        message = make_message(chat_id=333)
        self.assertFalse(_is_admin(message, settings))


class FormatEscalationNoticeTest(unittest.TestCase):
    def test_includes_username_when_present(self):
        message = make_message(chat_id=555, username="ivan", full_name="Иван Иванов")
        notice = _format_escalation_notice(message, "Делаете ли вы татуировки?")
        self.assertIn("@ivan", notice)
        self.assertIn("Иван Иванов", notice)
        self.assertIn("chat_id=555", notice)
        self.assertIn("Делаете ли вы татуировки?", notice)

    def test_handles_missing_username(self):
        message = make_message(chat_id=555, username=None, full_name="Без ника")
        notice = _format_escalation_notice(message, "Вопрос без ответа")
        self.assertIn("без username", notice)
        self.assertIn("Без ника", notice)

    def test_handles_missing_from_user(self):
        message = SimpleNamespace(
            chat=SimpleNamespace(id=555), from_user=None
        )
        notice = _format_escalation_notice(message, "Вопрос")
        self.assertIn("неизвестно", notice)
        self.assertIn("без username", notice)


class IsPlainGreetingTest(unittest.TestCase):
    def test_recognizes_common_greetings(self):
        for text in [
            "привет", "Привет", "ПРИВЕТ", "привет!", "здравствуйте",
            "добрый день", "добрый вечер", "hi", "hello", "ку",
        ]:
            with self.subTest(text=text):
                self.assertTrue(_is_plain_greeting(text))

    def test_greeting_with_surrounding_whitespace(self):
        self.assertTrue(_is_plain_greeting("  привет  "))

    def test_does_not_match_question_starting_with_greeting(self):
        self.assertFalse(
            _is_plain_greeting("привет, а сколько стоит доставка?")
        )

    def test_does_not_match_real_question(self):
        self.assertFalse(_is_plain_greeting("какой у вас режим работы?"))

    def test_empty_string_is_not_a_greeting(self):
        self.assertFalse(_is_plain_greeting(""))


if __name__ == "__main__":
    unittest.main()

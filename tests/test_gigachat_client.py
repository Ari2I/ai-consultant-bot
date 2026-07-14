"""
Тесты для llm.gigachat_client.GigaChatClient.

Реальные вызовы к GigaChat API не выполняются — вместо настоящего
gigachat.GigaChat передаётся фиктивный клиент (параметр `client`),
что позволяет проверить логику ретраев и обработку ошибок без сети,
без OAuth и без сертификата Минцифры.
"""

import unittest
from unittest.mock import MagicMock, patch

from gigachat.exceptions import (
    AuthenticationError,
    RateLimitError,
    ServerError,
)

from llm.gigachat_client import GigaChatApiError, GigaChatClient

_FAKE_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"


def _rate_limit_error() -> RateLimitError:
    return RateLimitError(_FAKE_URL, 429, b"", None)


def _server_error() -> ServerError:
    return ServerError(_FAKE_URL, 500, b"", None)


def _auth_error() -> AuthenticationError:
    return AuthenticationError(_FAKE_URL, 401, b"", None)


def _make_success_response(text: str):
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


class GigaChatClientSuccessTest(unittest.TestCase):
    def test_generate_answer_returns_model_text(self):
        fake_client = MagicMock()
        fake_client.chat.return_value = _make_success_response(
            "Магазин работает с 10:00 до 20:00."
        )
        client = GigaChatClient(
            credentials="test-credentials",
            scope="GIGACHAT_API_PERS",
            model="GigaChat-2",
            client=fake_client,
        )

        answer = client.generate_answer(
            question="Когда вы работаете?", context="Режим работы: 10:00-20:00"
        )

        self.assertEqual(answer, "Магазин работает с 10:00 до 20:00.")
        fake_client.chat.assert_called_once()
        payload = fake_client.chat.call_args.args[0]
        self.assertEqual(payload.model, "GigaChat-2")
        self.assertEqual(len(payload.messages), 2)
        self.assertEqual(payload.messages[0].role, "system")
        self.assertEqual(payload.messages[1].role, "user")


class GigaChatClientRetryTest(unittest.TestCase):
    @patch("llm.gigachat_client.time.sleep")
    def test_retries_on_rate_limit_then_succeeds(self, mock_sleep):
        fake_client = MagicMock()
        fake_client.chat.side_effect = [
            _rate_limit_error(),
            _make_success_response("Ответ после повтора"),
        ]
        client = GigaChatClient(
            credentials="k",
            scope="GIGACHAT_API_PERS",
            model="GigaChat-2",
            max_retries=2,
            retry_backoff_seconds=0.01,
            client=fake_client,
        )

        answer = client.generate_answer("вопрос", "контекст")

        self.assertEqual(answer, "Ответ после повтора")
        self.assertEqual(fake_client.chat.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("llm.gigachat_client.time.sleep")
    def test_raises_after_exhausting_retries(self, mock_sleep):
        fake_client = MagicMock()
        fake_client.chat.side_effect = _server_error()
        client = GigaChatClient(
            credentials="k",
            scope="GIGACHAT_API_PERS",
            model="GigaChat-2",
            max_retries=2,
            retry_backoff_seconds=0.01,
            client=fake_client,
        )

        with self.assertRaises(GigaChatApiError):
            client.generate_answer("вопрос", "контекст")

        # 1 первая попытка + 2 повтора = 3 вызова
        self.assertEqual(fake_client.chat.call_count, 3)


class GigaChatClientNonRetryableTest(unittest.TestCase):
    @patch("llm.gigachat_client.time.sleep")
    def test_authentication_error_is_not_retried(self, mock_sleep):
        fake_client = MagicMock()
        fake_client.chat.side_effect = _auth_error()
        client = GigaChatClient(
            credentials="wrong-credentials",
            scope="GIGACHAT_API_PERS",
            model="GigaChat-2",
            max_retries=2,
            retry_backoff_seconds=0.01,
            client=fake_client,
        )

        with self.assertRaises(GigaChatApiError):
            client.generate_answer("вопрос", "контекст")

        fake_client.chat.assert_called_once()
        mock_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()

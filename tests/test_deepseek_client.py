"""
Тесты для llm.deepseek_client.DeepSeekClient.

Реальные вызовы к DeepSeek API не выполняются — вместо настоящего
openai.OpenAI передаётся фиктивный клиент (параметр `client`), что
позволяет проверить логику ретраев и обработку ошибок без сети и без
расхода реальных токенов.
"""

import unittest
from unittest.mock import MagicMock, patch

import httpx
from openai import (
    APIConnectionError,
    AuthenticationError,
    InternalServerError,
    RateLimitError,
)

from llm.deepseek_client import DeepSeekApiError, DeepSeekClient

_FAKE_REQUEST = httpx.Request(
    "POST", "https://api.deepseek.com/chat/completions"
)


def _fake_response(status_code: int) -> httpx.Response:
    return httpx.Response(status_code, request=_FAKE_REQUEST)


def _make_success_response(text: str):
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


class DeepSeekClientSuccessTest(unittest.TestCase):
    def test_generate_answer_returns_model_text(self):
        fake_client = MagicMock()
        fake_client.chat.completions.create.return_value = (
            _make_success_response("Магазин работает с 10:00 до 20:00.")
        )
        client = DeepSeekClient(
            api_key="test-key",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            client=fake_client,
        )

        answer = client.generate_answer(
            question="Когда вы работаете?", context="Режим работы: 10:00-20:00"
        )

        self.assertEqual(answer, "Магазин работает с 10:00 до 20:00.")
        fake_client.chat.completions.create.assert_called_once()
        call_kwargs = fake_client.chat.completions.create.call_args.kwargs
        self.assertEqual(call_kwargs["model"], "deepseek-v4-flash")
        self.assertEqual(call_kwargs["messages"][0]["role"], "system")
        self.assertEqual(call_kwargs["messages"][1]["role"], "user")


class DeepSeekClientRetryTest(unittest.TestCase):
    @patch("llm.deepseek_client.time.sleep")
    def test_retries_on_rate_limit_then_succeeds(self, mock_sleep):
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [
            RateLimitError(
                "too many requests",
                response=_fake_response(429),
                body=None,
            ),
            _make_success_response("Ответ после повтора"),
        ]
        client = DeepSeekClient(
            api_key="k",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            max_retries=2,
            retry_backoff_seconds=0.01,
            client=fake_client,
        )

        answer = client.generate_answer("вопрос", "контекст")

        self.assertEqual(answer, "Ответ после повтора")
        self.assertEqual(fake_client.chat.completions.create.call_count, 2)
        mock_sleep.assert_called_once()

    @patch("llm.deepseek_client.time.sleep")
    def test_raises_after_exhausting_retries(self, mock_sleep):
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = InternalServerError(
            "server error", response=_fake_response(500), body=None
        )
        client = DeepSeekClient(
            api_key="k",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            max_retries=2,
            retry_backoff_seconds=0.01,
            client=fake_client,
        )

        with self.assertRaises(DeepSeekApiError):
            client.generate_answer("вопрос", "контекст")

        # 1 первая попытка + 2 повтора = 3 вызова
        self.assertEqual(fake_client.chat.completions.create.call_count, 3)

    @patch("llm.deepseek_client.time.sleep")
    def test_connection_error_is_retried(self, mock_sleep):
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = [
            APIConnectionError(request=_FAKE_REQUEST),
            _make_success_response("Ответ после обрыва соединения"),
        ]
        client = DeepSeekClient(
            api_key="k",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            max_retries=2,
            retry_backoff_seconds=0.01,
            client=fake_client,
        )

        answer = client.generate_answer("вопрос", "контекст")

        self.assertEqual(answer, "Ответ после обрыва соединения")


class DeepSeekClientNonRetryableTest(unittest.TestCase):
    @patch("llm.deepseek_client.time.sleep")
    def test_authentication_error_is_not_retried(self, mock_sleep):
        fake_client = MagicMock()
        fake_client.chat.completions.create.side_effect = AuthenticationError(
            "invalid api key", response=_fake_response(401), body=None
        )
        client = DeepSeekClient(
            api_key="wrong-key",
            base_url="https://api.deepseek.com",
            model="deepseek-v4-flash",
            max_retries=2,
            retry_backoff_seconds=0.01,
            client=fake_client,
        )

        with self.assertRaises(DeepSeekApiError):
            client.generate_answer("вопрос", "контекст")

        fake_client.chat.completions.create.assert_called_once()
        mock_sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()

"""Тесты для bot.rate_limiter.RateLimiter."""

import unittest
from unittest.mock import patch

from bot.rate_limiter import RateLimiter


class RateLimiterTest(unittest.TestCase):
    def test_allows_up_to_limit(self):
        limiter = RateLimiter(max_per_minute=3)
        for _ in range(3):
            self.assertTrue(limiter.allow(chat_id=1))

    def test_blocks_after_limit_exceeded(self):
        limiter = RateLimiter(max_per_minute=2)
        self.assertTrue(limiter.allow(chat_id=1))
        self.assertTrue(limiter.allow(chat_id=1))
        self.assertFalse(limiter.allow(chat_id=1))

    def test_limits_are_independent_per_chat(self):
        limiter = RateLimiter(max_per_minute=1)
        self.assertTrue(limiter.allow(chat_id=1))
        self.assertFalse(limiter.allow(chat_id=1))
        self.assertTrue(limiter.allow(chat_id=2))

    @patch("bot.rate_limiter.time.monotonic")
    def test_old_timestamps_expire_after_window(self, mock_monotonic):
        limiter = RateLimiter(max_per_minute=1)

        mock_monotonic.return_value = 1000.0
        self.assertTrue(limiter.allow(chat_id=1))
        self.assertFalse(limiter.allow(chat_id=1))

        # Прошло больше 60 секунд — лимит должен обновиться.
        mock_monotonic.return_value = 1061.0
        self.assertTrue(limiter.allow(chat_id=1))


if __name__ == "__main__":
    unittest.main()

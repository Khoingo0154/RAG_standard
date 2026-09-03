import asyncio
from unittest.mock import AsyncMock, patch

from telegram.error import TimedOut

from telegram_bot.main import send_reply


def test_send_reply_retries_network_timeout():
    message = AsyncMock()
    message.reply_text.side_effect = [TimedOut(), None]

    async def run_test():
        with patch("telegram_bot.main.asyncio.sleep", new=AsyncMock()) as sleep:
            sent = await send_reply(message, "answer")
        return sent, sleep

    sent, sleep = asyncio.run(run_test())

    assert sent is True
    assert message.reply_text.await_count == 2
    sleep.assert_awaited_once_with(1)

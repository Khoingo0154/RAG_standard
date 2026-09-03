"""Telegram bot: dùng chung retrieval pipeline, không cần FastAPI đang chạy."""

import asyncio
import logging

from telegram import Update
from telegram.error import NetworkError, TimedOut
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters

from retrieval.pipeline import search_and_generate
from shared.config import settings

logging.basicConfig(
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)
# httpx ghi URL đầy đủ ở mức INFO; URL Telegram có bot token nên phải tắt log này.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def selected_file_id(context: ContextTypes.DEFAULT_TYPE) -> str | None:
    """Ưu tiên file_id người dùng chọn trong chat, sau đó mới dùng mặc định."""
    return context.chat_data.get("file_id") or settings.TELEGRAM_DEFAULT_FILE_ID


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.effective_message.reply_text(
        "Chào bạn! Đây là bot hỏi đáp tài liệu RAG.\n\n"
        "1. Chọn tài liệu: /use <file_id>\n"
        "2. Gửi câu hỏi bình thường.\n"
        "Xem file đang chọn: /file"
    )


async def use_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.effective_message.reply_text("Cú pháp: /use <file_id>")
        return
    file_id = context.args[0].strip()
    context.chat_data["file_id"] = file_id
    await update.effective_message.reply_text(f"Đã chọn tài liệu: {file_id}")


async def file_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    file_id = selected_file_id(context)
    message = f"Tài liệu hiện tại: {file_id}" if file_id else "Bạn chưa chọn tài liệu. Dùng /use <file_id>."
    await update.effective_message.reply_text(message)


async def send_reply(message, text: str) -> bool:
    """Gửi câu trả lời, retry lỗi mạng tạm thời mà không lộ token vào log."""
    for attempt in range(3):
        try:
            await message.reply_text(text)
            return True
        except (TimedOut, NetworkError) as error:
            if attempt == 2:
                logger.error("Telegram send failed after retries: %s", type(error).__name__)
                return False
            delay = 2 ** attempt
            logger.warning("Telegram send timed out; retrying in %s second(s)", delay)
            await asyncio.sleep(delay)
    return False


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Chặn exception chưa xử lý để bot không in traceback dài hoặc lộ URL/token."""
    logger.error("Unhandled Telegram update error: %s", type(context.error).__name__)


async def answer_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    question = (update.effective_message.text or "").strip()
    file_id = selected_file_id(context)
    if not question:
        return
    if not file_id:
        await update.effective_message.reply_text("Bạn cần chọn tài liệu trước: /use <file_id>")
        return

    try:
        await update.effective_chat.send_action("typing")
    except (TimedOut, NetworkError):
        logger.warning("Telegram typing indicator timed out; continuing with query")
    try:
        result = await asyncio.to_thread(search_and_generate, question, file_id=file_id)
        # Telegram giới hạn 4096 ký tự mỗi message.
        for index in range(0, len(result.answer), 4096):
            if not await send_reply(update.effective_message, result.answer[index:index + 4096]):
                return
    except Exception as error:
        logger.exception("Telegram query failed")
        await send_reply(update.effective_message, "Không thể xử lý câu hỏi. Hãy thử lại sau ít phút.")


def main() -> None:
    if not settings.TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Thiếu TELEGRAM_BOT_TOKEN trong file .env")

    app = (
        ApplicationBuilder()
        .token(settings.TELEGRAM_BOT_TOKEN)
        .connect_timeout(settings.TELEGRAM_CONNECT_TIMEOUT)
        .read_timeout(settings.TELEGRAM_READ_TIMEOUT)
        .write_timeout(settings.TELEGRAM_WRITE_TIMEOUT)
        .pool_timeout(settings.TELEGRAM_POOL_TIMEOUT)
        .get_updates_connect_timeout(settings.TELEGRAM_CONNECT_TIMEOUT)
        .get_updates_read_timeout(settings.TELEGRAM_READ_TIMEOUT)
        .get_updates_write_timeout(settings.TELEGRAM_WRITE_TIMEOUT)
        .get_updates_pool_timeout(settings.TELEGRAM_POOL_TIMEOUT)
        .build()
    )
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("use", use_command))
    app.add_handler(CommandHandler("file", file_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, answer_message))
    app.add_error_handler(error_handler)
    logger.info("Telegram bot started with long polling")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()

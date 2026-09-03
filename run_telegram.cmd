@echo off
docker compose stop api >nul 2>&1
docker compose --profile telegram up -d
echo Telegram bot is starting. Check its logs with: docker compose logs -f telegram_bot

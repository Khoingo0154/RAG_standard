@echo off
docker compose stop telegram_bot >nul 2>&1
docker compose --profile swagger up -d
start http://127.0.0.1:8000/docs

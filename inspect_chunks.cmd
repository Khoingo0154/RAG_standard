@echo off
echo ========================================================
echo   DANG KIEM TRA SO LUONG KY TU, TU VA TOKEN CUA CHUNKS
echo ========================================================
docker compose --profile swagger run --rm api python inspect_chunks.py --file-id 8f525964-a864-43ef-b87b-34f6482d44f1
pause

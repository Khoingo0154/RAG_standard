@echo off
echo ========================================================
echo   DANG CHAY DANH GIA RETRIEVAL BENCHMARK (TOP-5)
echo ========================================================
docker compose --profile swagger run --rm -v "%cd%/evals:/app/evals" api python -m evals.retrieval_eval --file-id 8f525964-a864-43ef-b87b-34f6482d44f1 --top-k 5 --output evals/benchmark_baseline_law11_top5.json
echo.
echo Ket qua da duoc xuat tai: evals\benchmark_baseline_law11_top5.json
pause

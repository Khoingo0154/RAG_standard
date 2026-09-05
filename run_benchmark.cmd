@echo off
echo ========================================================
echo   DANG CHAY RAG BENCHMARK ENGINE (ADVANCED RAG)
echo ========================================================
docker compose --profile swagger run --rm -v "%cd%/evals:/app/evals" api python -m evals.benchmark --dataset evals/fifa_law11_basic.json --mode retrieval --top-k 5 --output-dir evals
echo.
echo ========================================================
echo   DA XUAT BAO CAO BENCHMARK TAI THU MUC evals/:
echo   - evals\benchmark_retrieval_fifa_law11_basic_top5.json
echo   - evals\benchmark_retrieval_fifa_law11_basic_top5.csv
echo   - evals\benchmark_retrieval_fifa_law11_basic_top5.md
echo ========================================================
pause

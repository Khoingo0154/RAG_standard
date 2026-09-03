@echo off
echo ========================================================
echo   DANG CHAY KIEM THU 20 CAU HOI RAG (FIFA LAWS OF THE GAME)
echo ========================================================
.venv\Scripts\python.exe -m evals.run_qa_batch
echo.
echo ========================================================
echo   DA XUAT KET QUA TAI:
echo   - evals\results_qa_testset.csv (Mo bang Excel)
echo   - evals\results_qa_testset.md  (Mo xem Markdown)
echo ========================================================
pause

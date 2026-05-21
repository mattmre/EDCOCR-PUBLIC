@echo off
cd /d "%~dp0"
echo Starting PDF Optimization (Compression)...
echo Target: /prepress (High Quality)
echo.
docker exec -it ocr_gpu_processor python3 /app/optimize_pdfs.py
echo.
echo Optimization complete.
pause

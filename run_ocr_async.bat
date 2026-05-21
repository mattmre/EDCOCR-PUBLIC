@echo off
cd /d "%~dp0"
echo Starting Industrial Async OCR Pipeline...
echo.
REM Stop and remove old container to clear any corrupted model caches
docker rm -f ocr_gpu_processor >nul 2>&1

REM Rebuild image before startup so code/model changes are baked in
docker-compose build
if errorlevel 1 (
  echo Docker build failed. Aborting.
  pause
  exit /b 1
)

REM Start the container (Command is now baked into docker-compose.yml)
docker-compose up -d

echo.
echo Container Started! Streaming logs (Ctrl+C to stop viewing logs, container will keep running)...
echo.
docker logs -f ocr_gpu_processor
pause

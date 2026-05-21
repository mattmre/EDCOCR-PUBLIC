@echo off
cd /d "%~dp0"
echo Building and Starting GPU OCR Container...
docker-compose up --build
pause

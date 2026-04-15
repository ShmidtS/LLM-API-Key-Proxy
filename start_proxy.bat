@echo off
chcp 65001 >nul
setlocal

echo ========================================
echo LLM API Key Proxy - Запуск
echo ========================================
echo.

cd /d "%~dp0"

REM Disable aiodns to fix "Domain name not found" errors when ping works
REM This must be set BEFORE Python imports aiohttp
set AIOHTTP_NO_EXTENSIONS=1

echo.
if "%PROXY_HOST%"=="" set PROXY_HOST=127.0.0.1
if "%PROXY_PORT%"=="" set PROXY_PORT=8000

echo Запуск прокси-сервера на http://%PROXY_HOST%:%PROXY_PORT%
echo.
echo.
echo Для остановки нажмите Ctrl+C
echo ========================================
echo.

python src/proxy_app/main.py --host %PROXY_HOST% --port %PROXY_PORT% 2>nul || py src/proxy_app/main.py --host %PROXY_HOST% --port %PROXY_PORT%

pause

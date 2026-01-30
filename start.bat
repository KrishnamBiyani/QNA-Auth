@echo off
echo Starting QNA-Auth System
echo.

echo Starting Backend Server...
start "QNA-Auth Backend" cmd /k "cd /d %~dp0 && call .venv\Scripts\activate.bat && python server/app.py"

timeout /t 3 /nobreak >nul

echo Starting Frontend Server...
start "QNA-Auth Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Both servers started!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
pause

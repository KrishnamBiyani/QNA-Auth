@echo off
REM QNA-Auth Setup Script for Windows
REM This script sets up the QNA-Auth environment

echo ==========================================
echo QNA-Auth Setup Script
echo ==========================================
echo.

REM Check Python version
echo Checking Python version...
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Install Python dependencies
echo Installing Python dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
echo.

REM Create necessary directories
echo Creating directories...
if not exist "dataset\samples" mkdir dataset\samples
if not exist "auth\device_embeddings" mkdir auth\device_embeddings
if not exist "model\checkpoints" mkdir model\checkpoints
echo.

REM Copy example config
echo Copying example configuration...
if not exist "config.py" (
    copy config.example.py config.py
    echo Config.py created - please review and adjust settings
) else (
    echo config.py already exists
)
echo.

REM Test installations
echo Testing installations...
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
echo.

echo ==========================================
echo Backend setup complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Review and adjust settings in config.py
echo 2. Start backend: python server\app.py
echo 3. Setup frontend: cd frontend ^&^& npm install
echo 4. Start frontend: cd frontend ^&^& npm run dev
echo.
pause

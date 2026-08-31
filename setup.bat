@echo off
setlocal
cd /d "%~dp0"

echo ==============================================================
echo  Local RAG Simple - first time setup
echo ==============================================================

where py >nul 2>nul
if not errorlevel 1 (
    set "PY=py"
) else (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python was not found.
        echo Install Python 3.11 or newer, then run this file again.
        pause
        exit /b 1
    )
    set "PY=python"
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment...
    %PY% -m venv .venv
    if errorlevel 1 goto :error
) else (
    echo [1/3] Virtual environment already exists.
)

echo [2/3] Updating pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error

echo [3/3] Installing project dependencies...
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error

echo.
echo Setup complete.
echo.
echo For the command-line interface:
echo Put your documents in the docs folder, then run run.bat
echo.
echo For the web interface:
echo Run run_ui.bat first. You can add and remove documents later from the UI.
echo.
pause
exit /b 0

:error
echo.
echo Setup failed. See the error above.
pause
exit /b 1

@echo off
setlocal
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" main.py
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        py main.py
    ) else (
        python main.py
    )
)

if errorlevel 1 (
    echo.
    echo The application stopped with an error.
)
pause

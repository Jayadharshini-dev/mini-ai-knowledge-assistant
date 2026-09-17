@echo off
if "%1"=="dev" (
    python -m uvicorn backend.app.main:app --reload --port 8000
) else if "%1"=="test" (
    python -m pytest -v
) else if "%1"=="lint" (
    python -m ruff check .
) else (
    echo Usage: make [dev^|test^|lint]
)

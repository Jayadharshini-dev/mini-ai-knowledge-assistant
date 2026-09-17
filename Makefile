.PHONY: dev test lint clean

dev:
	uvicorn backend.app.main:app --reload --port 8000

test:
	pytest -v

lint:
	ruff check .

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +

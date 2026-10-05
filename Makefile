.PHONY: install test test-network ruff ty

install:
	uv sync --all-groups --all-extras

test:
	uv run pytest

test-network:
	uv run pytest -m network

ruff:
	uv run ruff check --fix --unsafe-fixes --extend-select I
	uv run ruff format

ty:
	uv run ty check

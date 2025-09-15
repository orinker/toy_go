.PHONY: lint format test

lint:
	python -m ruff check .

format:
	python -m ruff check . --fix
	python -m black .

test:
	pytest -q


HEIGHT ?= 178
SIDE ?= left

.PHONY: run test test_coverage lint format

run:
	uv run python main.py --model_type blazepose --side $(SIDE) --runner_height $(HEIGHT)

test:
	uv run pytest tests/ -v

test_coverage:
	uv run pytest tests/ --cov=. --cov-report=term-missing --cov-config=pyproject.toml

lint:
	uv run ruff check .

format:
	uv run ruff format .

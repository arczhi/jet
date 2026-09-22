UV ?= uv
PY := $(UV) run

.PHONY: install fmt lint typecheck test ci smoke run app doctor clean

install:
	$(UV) sync

fmt:
	$(PY) ruff format src tests
	$(PY) ruff check --fix src tests

lint:
	$(PY) ruff check src tests

typecheck:
	$(PY) mypy

test:
	$(PY) pytest

ci: lint typecheck test

smoke:
	$(PY) python scripts/smoke.py

demo:
	chmod +x scripts/record-demo.sh && ./scripts/record-demo.sh

run:
	$(PY) jet

app:
	$(PY) jet app

doctor:
	$(PY) jet doctor

clean:
	rm -rf .venv .pytest_cache .mypy_cache .ruff_cache .coverage dist build jet.egg-info

PYTHON ?= .venv/bin/python

.PHONY: test-unit test-integration test-e2e test-full lint build-wheel demo-data demo

test-unit:
	$(PYTHON) -m pytest -q

test-integration:
	$(PYTHON) -m pytest -q -m "integration and not e2e and not paid and not network"

test-e2e:
	$(PYTHON) -m pytest -q -m "e2e and not paid and not network"

test-full:
	$(PYTHON) -m pytest -q -m "not paid and not network"

lint:
	uvx ruff check .

build-wheel:
	$(PYTHON) -m pip wheel . --no-deps -w dist

demo-data:
	$(PYTHON) examples/build_demo_data.py --output data/demo

demo:
	QUANT_DATA_DIR=data/demo $(PYTHON) examples/run_demo.py


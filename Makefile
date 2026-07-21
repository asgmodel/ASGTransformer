.PHONY: install test lint compile export verify

install:
	python -m pip install -e ".[dev,train]"

test:
	pytest -q

lint:
	ruff check src tests scripts examples

compile:
	python -m compileall -q src scripts examples

export:
	python scripts/export_transformers_model.py \
		--base-model $(BASE_MODEL) \
		--knowledge-dir data/processed \
		--output-dir dist/ASGTransformer

verify:
	python scripts/verify_transformers_checkpoint.py dist/ASGTransformer

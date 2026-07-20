.PHONY: install test lint serve train evaluate docker
install:
	pip install -e ".[dev,train]"
test:
	pytest -q --cov=asg_transformer
lint:
	ruff check src tests
serve:
	uvicorn asg_transformer.api.main:app --host 0.0.0.0 --port 8000 --reload
train:
	python -m asg_transformer.training.train_encoder --epochs 10 --batch-size 16
evaluate:
	python -m asg_transformer.evaluation.evaluate --top-k 5
docker:
	docker compose up --build

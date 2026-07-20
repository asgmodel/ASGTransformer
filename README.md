# ASG Transformer Professional

Enterprise-ready Transformer framework for semantic cyber-threat classification and tactic-aware scenario generation. The project combines Sentence Transformers, an optional cross-encoder reranker, ATT&CK-style knowledge catalogs, beam-search planning, FastAPI, structured logging, runtime monitoring, offline evaluation, Docker, CI/CD, and optional Weights & Biases experiment tracking.

> Use only in authorized defensive research, cyber-range training, security validation, and threat-intelligence workflows.

## Core capabilities

- Semantic retrieval for techniques, software, and threat groups.
- Tactic-aware cyber scenario generation using beam search.
- Weighted semantic and transition scoring.
- Optional cross-encoder reranking.
- Fine-tuning pipeline based on contrastive learning.
- Recall@K, MRR, and latency evaluation.
- W&B experiment tracking and model artifacts.
- FastAPI endpoints with Swagger and ReDoc.
- Runtime request, error, and latency monitoring.
- Structured JSON logging.
- CLI, Docker, Compose, Makefile, tests, and GitHub Actions.

## Architecture

```text
Client / CLI
     │
     ▼
FastAPI + Validation + Observability
     │
     ▼
ASGTransformerService
     ├── KnowledgeCatalog
     ├── SemanticEncoder
     │     ├── Sentence Transformer
     │     └── Optional Cross Encoder
     └── TransformerScenarioGenerator
           ├── Tactic Ordering
           ├── Transition Matrix
           └── Beam Search
     │
     ▼
Ranked Intelligence / Ordered Scenario
```

## Project structure

```text
ASGTransformer/
├── .github/workflows/ci.yml       # Automated linting and tests
├── data/processed/                # Techniques, tactics, groups, software, transitions
├── docs/ARCHITECTURE.md           # Architecture notes
├── examples/client.py             # Python API client
├── models/                        # Fine-tuned encoder and checkpoints
├── scripts/                       # API and training convenience scripts
├── src/asg_transformer/
│   ├── api/                       # FastAPI application
│   ├── cli/                       # `asg` command-line interface
│   ├── core/                      # Catalog, schemas, and service layer
│   ├── evaluation/                # Recall@K, MRR, and latency evaluation
│   ├── models/                    # Encoder and scenario generator
│   ├── monitoring/                # W&B integration
│   ├── observability/             # Structured logs and runtime metrics
│   ├── training/                  # Transformer fine-tuning
│   └── config.py                  # Environment-based configuration
├── tests/                         # Unit and API tests
├── .env.example
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── CONTRIBUTING.md
├── SECURITY.md
└── LICENSE
```

## Requirements

- Python 3.10–3.12 recommended
- pip 23+
- 4 GB RAM minimum
- NVIDIA GPU optional for faster training

## Local installation

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install the complete development and training environment:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev,train]"
cp .env.example .env
```

Windows:

```powershell
Copy-Item .env.example .env
```

## Google Colab installation

```python
%cd /content/ASGTransformer
!pip install --upgrade pip setuptools wheel
!pip install -e ".[dev,train]"
```

Confirm the project configuration:

```python
!asg doctor
```

## Configuration

The `.env` file is read from the project root. Relative paths are resolved from the project root, not from the current shell directory.

```env
ASG_DATA_DIR=data/processed
ASG_MODEL_DIR=models/asg-encoder
ASG_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
ASG_DEVICE=auto
ASG_TOP_K=5
ASG_MAX_SCENARIO_STEPS=8
ASG_MIN_CONFIDENCE=0.25
ASG_ENABLE_RERANKER=false
ASG_RERANKER_NAME=cross-encoder/ms-marco-MiniLM-L-6-v2
ASG_LOG_LEVEL=INFO

WANDB_API_KEY=
WANDB_PROJECT=asg-transformer
WANDB_ENTITY=
WANDB_MODE=online
```

## Run the API

```bash
asg serve --host 0.0.0.0 --port 8000
```

Development mode:

```bash
uvicorn asg_transformer.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Interfaces:

- Swagger: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Health: `http://localhost:8000/health`
- Runtime metrics: `http://localhost:8000/metrics`

## API usage

Classify techniques:

```bash
curl -X POST http://localhost:8000/v1/classify/technique \
  -H "Content-Type: application/json" \
  -d '{"text":"The adversary modifies controller logic and suppresses alarms","top_k":5}'
```

Generate a scenario:

```bash
curl -X POST http://localhost:8000/v1/scenarios/generate \
  -H "Content-Type: application/json" \
  -d '{"text":"Remote access to an industrial controller followed by unsafe process manipulation","max_steps":6,"beam_width":5,"transition_weight":0.35}'
```

Python client:

```bash
python examples/client.py
```

## CLI usage

```bash
asg doctor
asg predict "Credential theft followed by remote execution" --task technique --top-k 5
asg scenario "Initial access followed by persistence and lateral movement" --max-steps 8
asg serve --port 8000
```

## Training

Basic training:

```bash
python -m asg_transformer.training.train_encoder \
  --base-model sentence-transformers/all-MiniLM-L6-v2 \
  --data-dir data/processed \
  --output-dir models/asg-encoder \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 2e-5
```

The service automatically loads `models/asg-encoder` when it exists. Otherwise, it loads the configured base model.

### W&B monitoring

Authenticate:

```bash
wandb login
```

Train with online monitoring:

```bash
python -m asg_transformer.training.train_encoder \
  --wandb \
  --wandb-project asg-transformer \
  --wandb-run-name minilm-baseline \
  --epochs 10 \
  --batch-size 16
```

Offline mode:

```bash
python -m asg_transformer.training.train_encoder \
  --wandb \
  --wandb-mode offline \
  --epochs 10
```

Synchronize later:

```bash
wandb sync wandb/offline-run-*
```

Tracked information includes hyperparameters, dataset size, evaluation scores, training steps, runtime, CPU/GPU utilization collected by W&B, run status, and model artifacts.

## Evaluation

```bash
python -m asg_transformer.evaluation.evaluate \
  --data-dir data/processed \
  --model-dir models/asg-encoder \
  --top-k 5 \
  --output reports/evaluation.json
```

Reported metrics:

- Recall@K
- Mean Reciprocal Rank (MRR)
- Average inference latency
- P95 inference latency

## Testing and quality

```bash
pytest -q
pytest -q --cov=asg_transformer --cov-report=term-missing
ruff check src tests
```

Convenience commands:

```bash
make install
make test
make lint
make serve
make train
make evaluate
```

## Docker

```bash
docker build -t asg-transformer:3.0.0 .
docker run --rm -p 8000:8000 --env-file .env asg-transformer:3.0.0
```

Docker Compose:

```bash
docker compose up --build
```

## Runtime monitoring

`GET /metrics` returns in-process operational information:

```json
{
  "requests": 120,
  "server_errors": 0,
  "latency_ms": {
    "avg": 42.3,
    "p50": 31.0,
    "p95": 88.5,
    "p99": 120.2
  },
  "routes": {
    "/health": 20,
    "/v1/scenarios/generate": 100
  }
}
```

For multi-worker production deployments, connect the service to an external metrics platform such as Prometheus rather than aggregating only in process memory.

## Data files

The project expects these files under `ASG_DATA_DIR`:

```text
techniques.json
software.json
groups.json
tactics.json
technique_to_tactic.json
transition_scores.json
```

The application fails fast with the absolute missing path when required data is unavailable.

## Production recommendations

- Pin and scan container images.
- Keep W&B projects private when experiment data is sensitive.
- Never commit API keys or confidential threat-intelligence records.
- Place the API behind authentication, TLS, rate limiting, and an API gateway.
- Use an external model registry and object storage for large model artifacts.
- Add domain-specific labeled examples before treating scores as production-grade probabilities.

## Current validation

The source tree compiles successfully. Core catalog, path resolution, scenario logic, and runtime metrics tests pass. Full API and model-loading tests require the project dependencies and access to the configured Sentence Transformer model.

## License

MIT. See `LICENSE`.

# ASGTransformer

A production-oriented, Hugging Face-ready defensive cybersecurity model that accepts text and returns a professional, grounded scenario as text and structured JSON.

## Unified Pipeline

```text
Input Text
   │
   ▼
Semantic Encoder
   │  retrieves grounded techniques, software, and groups
   ▼
Scenario Generator
   │  applies tactic ordering, transition scoring, and beam search
   ▼
Duration Planner
   │  assigns the stage budget and estimated duration
   ▼
Grounded Text Generator
   │  converts the plan into professional English or Arabic text
   ▼
Text + JSON Output
```

## Key Capabilities

- One deployable model interface: `ASGTransformer`
- Text-to-text scenario generation
- Catalog-grounded semantic retrieval
- Beam-search scenario planning
- Duration planning for each stage
- English and Arabic output
- Structured output for APIs and dashboards
- Hugging Face-style `save_pretrained`, `from_pretrained`, and `push_to_hub`
- FastAPI, CLI, Docker, tests, evaluation, and W&B monitoring
- Bundled knowledge catalog inside the exported model repository

## Installation

### Local

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -e ".[dev,train]"
```

### Google Colab

```python
%cd /content/ASG-Transformer-Professional
!pip install --upgrade pip setuptools wheel
!pip install -e ".[dev,train]"
```

## Configuration

Create `.env` in the project root:

```env
ASG_DATA_DIR=data/processed
ASG_MODEL_DIR=models/asg-encoder
ASG_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
ASG_DEVICE=auto
```

Relative paths are resolved from the project root, not the current shell directory.

## Train the Semantic Encoder

```bash
python -m asg_transformer.training.train_encoder \
  --base-model sentence-transformers/all-MiniLM-L6-v2 \
  --output-dir models/asg-encoder \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 2e-5
```

### Train with W&B

```bash
wandb login

python -m asg_transformer.training.train_encoder \
  --base-model sentence-transformers/all-MiniLM-L6-v2 \
  --output-dir models/asg-encoder \
  --epochs 10 \
  --batch-size 16 \
  --learning-rate 2e-5 \
  --wandb \
  --wandb-project asg-unified-transformer \
  --wandb-run-name encoder-v1
```

## Use the Unified Model Locally

```python
from asg_transformer.config import settings
from asg_transformer.core.catalog import KnowledgeCatalog
from asg_transformer.models.semantic_encoder import SemanticEncoder
from asg_transformer import ASGTransformer

catalog = KnowledgeCatalog(settings.data_dir)
encoder = SemanticEncoder(
    model_name=settings.model_name,
    model_dir=str(settings.model_dir),
    device=settings.device,
)
model = ASGTransformer(catalog, encoder)

result = model.generate(
    "Create a defensive enterprise scenario focused on phishing and credential access.",
    max_steps=6,
    total_duration_minutes=240,
    language="en",
)

print(result.generated_text)
print(result.to_dict())
```

## Export as One Hugging Face Model Package

```bash
python scripts/export_huggingface.py \
  --output-dir dist/ASGTransformer
```

The exported directory contains everything required by the unified model:

```text
dist/ASGTransformer/
├── asg_config.json
├── README.md
├── encoder/
│   ├── config.json
│   ├── model.safetensors
│   ├── modules.json
│   ├── tokenizer.json
│   └── 1_Pooling/config.json
└── knowledge/
    ├── techniques.json
    ├── tactics.json
    ├── technique_to_tactic.json
    ├── transition_scores.json
    ├── software.json
    └── groups.json
```

## Publish to Hugging Face

Log in from Colab or a terminal:

```bash
hf auth login
```

Publish the unified package:

```bash
python scripts/export_huggingface.py \
  --output-dir dist/ASGTransformer \
  --repo-id asgmodel/ASGTransformer
```

Or from Python:

```python
url = model.push_to_hub(
    repo_id="asgmodel/ASGTransformer",
    private=False,
)
print(url)
```

## Load from Hugging Face

```python
from asg_transformer import ASGTransformer

model = ASGTransformer.from_pretrained(
    "asgmodel/ASGTransformer"
)

result = model.generate(
    "Generate an authorized defensive scenario for an enterprise Windows environment.",
    language="en",
)

print(result.generated_text)
```

Arabic output:

```python
result = model.generate(
    "أنشئ سيناريو تدريبي دفاعي للتصيد والوصول إلى بيانات الاعتماد.",
    language="ar",
)
print(result.generated_text)
```

## API

```bash
uvicorn asg_transformer.api.main:app --host 0.0.0.0 --port 8000
```

Swagger:

```text
http://localhost:8000/docs
```

Request:

```bash
curl -X POST http://localhost:8000/v1/scenarios/generate \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Create a defensive phishing and credential access simulation",
    "max_steps": 6,
    "beam_width": 5,
    "transition_weight": 0.35,
    "total_duration_minutes": 240,
    "language": "en"
  }'
```

## Testing

```bash
pytest -q
pytest --cov=asg_transformer
```

## Architecture

```text
src/asg_transformer/
├── api/                    FastAPI endpoints
├── cli/                    Command-line interface
├── core/                   Catalog, schemas, and service layer
├── evaluation/             Retrieval and latency evaluation
├── models/
│   ├── semantic_encoder.py Semantic retrieval encoder
│   ├── scenario_generator.py Beam-search scenario planner
│   ├── duration_planner.py Stage duration planning
│   ├── text_generator.py   Professional grounded renderer
│   └── unified_model.py    Single Hugging Face-ready interface
├── monitoring/             W&B experiment tracking
├── observability/          Logs and runtime metrics
└── training/               Encoder training pipeline
```

## Important Design Note

`ASGTransformer` is the deployable model. `TransformerScenarioGenerator` is an internal planning component. The exported Hugging Face repository bundles the encoder, catalog, planner configuration, duration settings, and generation logic behind one public interface.

## Intended Use

This project is intended for authorized defensive cybersecurity training, tabletop exercises, security-control validation, detection engineering, and incident-response preparation.

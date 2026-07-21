# Hugging Face Hub Deployment

## Build

```bash
python scripts/export_transformers_model.py \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --knowledge-dir data/processed \
  --output-dir dist/ASGTransformer
```

## Verify

```bash
python scripts/verify_transformers_checkpoint.py dist/ASGTransformer
```

## Authenticate

```bash
hf auth login
```

## Publish

```bash
python scripts/export_transformers_model.py \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --knowledge-dir data/processed \
  --output-dir dist/ASGTransformer \
  --repo-id asgmodel/ASGTransformer
```

## Load

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "asgmodel/ASGTransformer"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)
```

Custom model code requires `trust_remote_code=True`. Review the repository code before enabling it.

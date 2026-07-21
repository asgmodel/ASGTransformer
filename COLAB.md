# Google Colab Workflow

## Install

```python
!git clone YOUR_REPOSITORY_URL ASGTransformer
%cd /content/ASGTransformer
!python -m pip install --upgrade pip setuptools wheel
!pip install -e ".[dev,train]"
```

## Authenticate with Hugging Face

```python
from huggingface_hub import notebook_login
notebook_login()
```

## Build the model repository

```python
!python scripts/export_transformers_model.py \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --knowledge-dir data/processed \
  --output-dir dist/ASGTransformer
```

## Verify

```python
!python scripts/verify_transformers_checkpoint.py dist/ASGTransformer
```

## Upload

```python
!python scripts/export_transformers_model.py \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --knowledge-dir data/processed \
  --output-dir dist/ASGTransformer \
  --repo-id asgmodel/ASGTransformer
```

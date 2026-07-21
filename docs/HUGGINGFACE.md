# Hugging Face Deployment

## 1. Train the encoder

```bash
python -m asg_transformer.training.train_encoder \
  --output-dir models/asg-encoder \
  --epochs 10 \
  --batch-size 16
```

## 2. Export the unified package

```bash
python scripts/export_huggingface.py \
  --output-dir dist/ASGTransformer
```

The package contains the encoder weights, the knowledge catalog, unified model configuration, model card, dependency file, and inference handler.

## 3. Authenticate

```bash
hf auth login
```

## 4. Publish

```bash
python scripts/export_huggingface.py \
  --output-dir dist/ASGTransformer \
  --repo-id asgmodel/ASGTransformer
```

## 5. Load

```python
from asg_transformer import ASGTransformer

model = ASGTransformer.from_pretrained("asgmodel/ASGTransformer")
result = model.generate("Create a defensive scenario for credential access.")
print(result.generated_text)
```

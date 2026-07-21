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
  --output-dir dist/ASG-Unified-Scenario-Model
```

The package contains the encoder weights, the knowledge catalog, unified model configuration, model card, dependency file, and inference handler.

## 3. Authenticate

```bash
hf auth login
```

## 4. Publish

```bash
python scripts/export_huggingface.py \
  --output-dir dist/ASG-Unified-Scenario-Model \
  --repo-id asgmodel/ASG-Unified-Scenario-Model
```

## 5. Load

```python
from asg_transformer import ASGUnifiedModel

model = ASGUnifiedModel.from_pretrained("asgmodel/ASG-Unified-Scenario-Model")
result = model.generate("Create a defensive scenario for credential access.")
print(result.generated_text)
```

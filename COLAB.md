# Google Colab Quick Start

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%cd /content/ASG-Transformer-Professional
!pip install --upgrade pip setuptools wheel
!pip install -e ".[dev,train]"
```

```python
!python -m asg_transformer.training.train_encoder \
  --output-dir models/asg-encoder \
  --epochs 10 \
  --batch-size 16 \
  --wandb \
  --wandb-project asg-unified-transformer
```

```python
from huggingface_hub import notebook_login
notebook_login()
```

```python
!python scripts/export_huggingface.py \
  --output-dir dist/ASGTransformer \
  --repo-id asgmodel/ASGTransformer
```

# ASGTransformer on Hugging Face Transformers

## Standard loading

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(
    "asgmodel/ASGTransformer",
    trust_remote_code=True,
)
model = AutoModelForCausalLM.from_pretrained(
    "asgmodel/ASGTransformer",
    trust_remote_code=True,
)
```

`trust_remote_code=True` is required because ASGTransformer is a custom
architecture not bundled inside the upstream Transformers package.

## Single checkpoint architecture

All trainable tensors are children of `ASGTransformerForCausalLM`:

- `generator.*`: complete causal language model
- `semantic_encoder.*`: semantic projection network
- `duration_head.*`: duration classifier
- `scenario_head.*`: scenario planning classifier

They are saved by `save_pretrained(..., safe_serialization=True)` in one
`model.safetensors` file unless automatic sharding is needed for a very large
checkpoint. Even when sharded, all shards remain part of the same model
repository and are loaded by one `from_pretrained()` call.

The tokenizer, configuration, generation configuration, custom Python model
code, prompt template, and knowledge assets are stored in the same repository.

## Export from a base model

```bash
python scripts/export_transformers_model.py \
  --base-model YOUR_BASE_CAUSAL_LM \
  --knowledge-dir data/processed \
  --output-dir dist/ASGTransformer
```

Upload:

```bash
huggingface-cli upload asgmodel/ASGTransformer dist/ASGTransformer .
```

# ASGTransformer Architecture

## Design objective

ASGTransformer provides one Hugging Face `PreTrainedModel` that combines text generation and auxiliary scenario intelligence. The architecture is intentionally compatible with `AutoModelForCausalLM` and stores every trainable tensor beneath one model checkpoint.

## Components

### Causal generator

`generator` is a complete model created through `AutoModelForCausalLM.from_config()`. During export, the selected base model's weights are copied into this submodule.

### Semantic encoder

The final hidden states are attention-mask pooled and projected into a normalized semantic vector. This representation supports retrieval, clustering, comparison, and optional semantic supervision.

### Duration head

The semantic vector is classified into configurable minute bins. The default bins are `5, 10, 15, 30, 45, 60, 90, 120`.

### Scenario head

The semantic vector is classified into configurable high-level scenario categories. Labels are persisted in `config.json`.

### Packaged knowledge

Knowledge JSON files are merged into `knowledge/catalog.json` during export. A deterministic retrieval layer selects context for prompt construction without requiring an additional model at inference time.

## Checkpoint layout

All parameters use one of these namespaces:

```text
generator.*
semantic_encoder.*
duration_head.*
scenario_head.*
```

Safetensors sharding does not change the logical model. One `from_pretrained()` call restores every shard and component.

## Standard loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
```

## Training objective

The model can combine causal language-model loss, cosine semantic loss, duration classification loss, and scenario classification loss. Each auxiliary contribution has a configurable weight.

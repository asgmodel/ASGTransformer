# ASGTransformer

**ASGTransformer** is a production-oriented, Hugging Face Transformers-compatible model architecture for authorized defensive cybersecurity scenario generation. It accepts a natural-language task and produces a professional scenario while exposing semantic representations, duration estimates, and scenario-type predictions from the same checkpoint.

The official model class is:

```python
ASGTransformerForCausalLM
```

It is designed to load through the standard Transformers interfaces:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
```

---

## 1. What makes ASGTransformer unified?

ASGTransformer packages all trainable components beneath one `PreTrainedModel`:

```text
ASGTransformerForCausalLM
│
├── generator.*
│   └── Complete causal language model
│
├── semantic_encoder.*
│   ├── Layer normalization
│   ├── Projection layers
│   ├── GELU activation
│   └── Normalized semantic embedding
│
├── duration_head.*
│   └── Duration-class prediction
│
└── scenario_head.*
    └── Scenario-category prediction
```

A single `save_pretrained()` call persists all weights into one logical Hugging Face checkpoint. Smaller checkpoints use `model.safetensors`; larger checkpoints may use multiple Safetensors shards plus an index file. In both cases, the entire architecture is restored with one `from_pretrained()` call.

The same model repository also contains:

- The tokenizer and tokenizer configuration.
- The custom model and configuration Python files.
- Generation configuration.
- The packaged cybersecurity knowledge catalog.
- The professional prompt template.
- The Hugging Face Model Card.

---

## 2. End-to-end architecture

```text
User text
   │
   ├──► Knowledge retrieval
   │       └── Matches packaged tactics, techniques, software, groups,
   │           mappings, and transition information
   │
   ├──► Grounded prompt construction
   │       └── Scope, safety rules, language, response structure,
   │           and selected catalog context
   │
   └──► Base causal language model
           │
           ├──► Hidden-state pooling
           │       └── Semantic encoder
           │              └── Normalized semantic embedding
           │
           ├──► Duration head
           │       └── Estimated scenario duration
           │
           ├──► Scenario head
           │       └── Scenario category
           │
           └──► Text generation
                   └── Professional defensive scenario
```

---

## 3. Main capabilities

- Native Hugging Face `PretrainedConfig` and `PreTrainedModel` architecture.
- Standard `AutoTokenizer` and `AutoModelForCausalLM` loading.
- Complete causal language-model weights stored with all ASG-specific heads.
- Semantic embedding extraction from the generator's hidden states.
- Configurable duration prediction bins.
- Configurable scenario categories.
- Packaged knowledge retrieval without requiring a second runtime model.
- Professional grounded prompt construction.
- English and Arabic response instructions.
- Standard Transformers `forward()` and `generate()` behavior.
- Convenience `generate_scenario()` interface.
- Multi-task training support.
- Safetensors export and checkpoint validation.
- Hugging Face Hub publication tooling.
- FastAPI, CLI, Docker, tests, and CI support.

---

## 4. Project structure

```text
ASGTransformer/
├── src/asg_transformer/
│   ├── hf/
│   │   ├── configuration_asg_transformer.py
│   │   └── modeling_asg_transformer.py
│   ├── api/
│   ├── cli/
│   ├── core/
│   ├── evaluation/
│   ├── monitoring/
│   ├── observability/
│   └── training/
├── data/processed/
│   ├── tactics.json
│   ├── techniques.json
│   ├── software.json
│   ├── groups.json
│   ├── technique_to_tactic.json
│   └── transition_scores.json
├── scripts/
│   ├── export_transformers_model.py
│   └── verify_transformers_checkpoint.py
├── examples/
│   └── huggingface_automodel.py
├── docs/
├── tests/
├── pyproject.toml
├── Dockerfile
└── README.md
```

---

## 5. Installation

### Development installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev,train]"
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev,train]"
```

### Runtime-only installation

```bash
pip install -e .
```

---

## 6. Build a complete ASGTransformer checkpoint

Choose a causal language model that is compatible with `AutoModelForCausalLM`.

```bash
python scripts/export_transformers_model.py \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --knowledge-dir data/processed \
  --output-dir dist/ASGTransformer
```

Useful export options:

```text
--semantic-dim       Semantic embedding dimension, default: 256
--duration-bins      Comma-separated minute values
--torch-dtype        auto, float32, float16, or bfloat16
--revision           Optional base-model revision
--max-shard-size     Maximum Safetensors shard size
--repo-id            Optional Hugging Face repository ID
--private            Create or upload to a private repository
--trust-remote-code  Permit custom code in the selected base model
```

Example with custom duration bins:

```bash
python scripts/export_transformers_model.py \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --duration-bins 15,30,45,60,90,120,180,240 \
  --semantic-dim 384 \
  --knowledge-dir data/processed \
  --output-dir dist/ASGTransformer
```

---

## 7. Exported Hugging Face repository

```text
dist/ASGTransformer/
├── config.json
├── generation_config.json
├── model.safetensors
│   └── or model-00001-of-0000N.safetensors + index
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
├── configuration_asg_transformer.py
├── modeling_asg_transformer.py
├── README.md
└── knowledge/
    ├── catalog.json
    ├── prompt_template.txt
    ├── tactics.json
    ├── techniques.json
    ├── software.json
    ├── groups.json
    ├── technique_to_tactic.json
    └── transition_scores.json
```

Checkpoint parameter namespaces:

```text
generator.*
semantic_encoder.*
duration_head.*
scenario_head.*
```

---

## 8. Verify the exported model

Run the complete validation script before publishing:

```bash
python scripts/verify_transformers_checkpoint.py dist/ASGTransformer
```

The verifier checks:

- Required repository files.
- Safetensors checkpoint presence and shard completeness.
- All four required parameter namespaces.
- `AutoConfig.from_pretrained()`.
- `AutoTokenizer.from_pretrained()`.
- `AutoModelForCausalLM.from_pretrained()`.
- Model class and `model_type` correctness.
- Forward-pass output shapes.
- Semantic embedding dimension.
- Text generation.

For large models, generation may be skipped during a structural-only check:

```bash
python scripts/verify_transformers_checkpoint.py \
  dist/ASGTransformer \
  --skip-generation
```

---

## 9. Load from Hugging Face

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "asgmodel/ASGTransformer"

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="auto",
)
```

`trust_remote_code=True` is required because ASGTransformer provides a custom model architecture. Use it only after reviewing and trusting the repository code.

---

## 10. Recommended text-to-text interface

```python
result = model.generate_scenario(
    tokenizer,
    (
        "Create an authorized defensive enterprise scenario focused on "
        "phishing awareness, credential protection, and response readiness."
    ),
    language="en",
    max_new_tokens=384,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
)

print(result["text"])
print(result["estimated_duration_minutes"])
print(result["scenario_type"])
```

Returned dictionary:

```python
{
    "text": str,
    "estimated_duration_minutes": int,
    "duration_probabilities": list[float],
    "scenario_type": str,
    "scenario_probabilities": list[float],
    "semantic_embedding": list[float],
    "prompt": str,
}
```

Arabic instruction example:

```python
result = model.generate_scenario(
    tokenizer,
    "أنشئ سيناريو تدريبي دفاعي للتوعية بالتصيد وحماية بيانات الاعتماد.",
    language="ar",
    max_new_tokens=384,
)

print(result["text"])
```

The selected base model must have sufficient Arabic capability for strong Arabic output quality.

---

## 11. Standard Transformers generation

```python
import torch

prompt = model.build_grounded_prompt(
    "Create a defensive credential-protection tabletop exercise.",
    language="en",
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.inference_mode():
    sequences = model.generate(
        **inputs,
        max_new_tokens=384,
        do_sample=False,
    )

completion_ids = sequences[:, inputs["input_ids"].shape[1]:]
completion = tokenizer.decode(completion_ids[0], skip_special_tokens=True)
print(completion)
```

---

## 12. Semantic encoder

```python
import torch

inputs = tokenizer(
    "Defensive phishing awareness and credential protection exercise",
    return_tensors="pt",
).to(model.device)

with torch.inference_mode():
    embedding = model.encode(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
    )

print(embedding.shape)
```

Embeddings are L2-normalized and may be used with a vector index or similarity search layer.

---

## 13. Duration and scenario prediction

```python
with torch.inference_mode():
    embedding = model.encode(**inputs)
    duration_logits, duration_minutes = model.predict_duration(embedding)
    scenario_logits, scenario_labels = model.predict_scenario_type(embedding)

print(int(duration_minutes[0]))
print(scenario_labels[0])
```

The prediction heads require task-specific training before their probabilities should be treated as calibrated estimates.

---

## 14. Multi-task training

The model supports language-model, semantic, duration, and scenario targets in the same forward pass:

```python
outputs = model(
    input_ids=input_ids,
    attention_mask=attention_mask,
    labels=language_model_labels,
    semantic_targets=semantic_targets,
    duration_labels=duration_labels,
    scenario_labels=scenario_labels,
)

loss = outputs.loss
loss.backward()
```

The total objective is:

```text
Language-model loss
+ semantic_loss_weight × semantic cosine loss
+ duration_loss_weight × duration cross-entropy
+ scenario_loss_weight × scenario cross-entropy
```

The weights are stored in `config.json` and may be changed through `ASGTransformerConfig`.

---

## 15. Publish to Hugging Face Hub

Authenticate:

```bash
hf auth login
```

Build and upload in one command:

```bash
python scripts/export_transformers_model.py \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --knowledge-dir data/processed \
  --output-dir dist/ASGTransformer \
  --repo-id asgmodel/ASGTransformer
```

Private repository:

```bash
python scripts/export_transformers_model.py \
  --base-model Qwen/Qwen2.5-0.5B-Instruct \
  --knowledge-dir data/processed \
  --output-dir dist/ASGTransformer \
  --repo-id asgmodel/ASGTransformer \
  --private
```

---

## 16. Tests and quality checks

```bash
pytest -q
ruff check src tests scripts examples
python -m compileall -q src scripts examples
```

Coverage:

```bash
pytest --cov=asg_transformer --cov-report=term-missing
```

---

## 17. Important model-quality note

Exporting a base causal language model into ASGTransformer initializes the ASG-specific semantic, duration, and scenario heads, but does not automatically train them. For production quality, fine-tune the complete model or the auxiliary heads using a reviewed dataset containing:

- Input task text.
- Professional target scenario text.
- Duration labels.
- Scenario-category labels.
- Optional semantic target vectors or positive/negative pairs.

The generated text quality also depends heavily on the selected base model, the fine-tuning dataset, decoding configuration, and human review process.

---

## 18. Safety and intended use

ASGTransformer is intended for:

- Authorized defensive cybersecurity training.
- Awareness exercises.
- Incident-response preparation.
- Executive and technical tabletop exercises.
- Architecture and control reviews.
- Detection-engineering planning.

It is not intended for unauthorized access, exploitation, malware development, evasion, credential theft, or instructions that facilitate real-world compromise. Generated material should be reviewed by qualified cybersecurity professionals before operational use.

---

## 19. License

This project is distributed under the MIT License. Base-model licenses and knowledge-source licenses remain independently applicable and must be reviewed before redistribution or commercial deployment.

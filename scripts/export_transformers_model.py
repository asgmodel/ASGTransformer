#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

from asg_transformer.hf import ASGTransformerConfig, ASGTransformerForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a complete ASGTransformer Hugging Face repository from a base "
            "causal language model."
        )
    )
    parser.add_argument("--base-model", required=True, help="Base causal LM ID or local path")
    parser.add_argument("--output-dir", default="dist/ASGTransformer")
    parser.add_argument("--knowledge-dir", default="data/processed")
    parser.add_argument("--semantic-dim", type=int, default=256)
    parser.add_argument("--duration-bins", default="5,10,15,30,45,60,90,120")
    parser.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def parse_duration_bins(raw: str) -> list[int]:
    try:
        bins = sorted({int(part.strip()) for part in raw.split(",") if part.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--duration-bins must contain comma-separated integers") from exc
    if not bins or any(value <= 0 for value in bins):
        raise argparse.ArgumentTypeError("--duration-bins must contain positive values")
    return bins


def resolve_dtype(name: str) -> str | torch.dtype:
    mapping: dict[str, str | torch.dtype] = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def copy_knowledge(source_dir: Path, output_dir: Path) -> None:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Knowledge directory does not exist: {source_dir}")

    target = output_dir / "knowledge"
    target.mkdir(parents=True, exist_ok=True)
    catalog: dict[str, Any] = {}

    json_files = sorted(source_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON knowledge files found in: {source_dir}")

    for source in json_files:
        destination = target / source.name
        shutil.copy2(source, destination)
        try:
            catalog[source.stem] = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON knowledge file: {source}") from exc

    (target / "catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (target / "prompt_template.txt").write_text(
        """You are ASGTransformer, a professional defensive cybersecurity scenario-generation model.
Create only authorized, high-level training content. Do not provide exploit code, payloads,
evasion procedures, credential theft steps, or instructions that enable real-world compromise.

Response language: {language}
User task:
{input}

Grounding context:
{knowledge}

Required response:
1. Scenario title
2. Executive summary
3. Assumptions and scope
4. Ordered defensive stages
5. Estimated duration
6. Monitoring and detection objectives
7. Expected learning outcomes

Professional response:
""",
        encoding="utf-8",
    )


def write_model_card(output_dir: Path, base_model: str) -> None:
    card = f'''---
library_name: transformers
pipeline_tag: text-generation
tags:
- custom-code
- causal-lm
- cybersecurity
- defensive-security
- scenario-generation
language:
- en
- ar
license: mit
base_model: {base_model}
---

# ASGTransformer

ASGTransformer is a custom Hugging Face Transformers architecture for authorized,
defensive cybersecurity scenario generation. It packages a complete causal language
model, semantic representation encoder, duration prediction head, scenario
classification head, tokenizer, prompt template, and knowledge catalog in one model
repository.

## Architecture

```text
Input text
   ├─► Packaged knowledge retrieval ─► Grounded prompt
   └─► Causal language model hidden states
             ├─► Semantic encoder
             ├─► Duration head
             ├─► Scenario head
             └─► Text generator
```

Every trainable parameter is saved under the same checkpoint namespace:

- `generator.*`
- `semantic_encoder.*`
- `duration_head.*`
- `scenario_head.*`

Large models may be automatically split into multiple Safetensors shards, but they
remain one logical model and load with one `from_pretrained()` call.

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

## Generate a professional scenario

```python
result = model.generate_scenario(
    tokenizer,
    "Create an authorized defensive phishing-awareness tabletop scenario.",
    language="en",
    max_new_tokens=384,
)

print(result["text"])
print(result["estimated_duration_minutes"])
print(result["scenario_type"])
```

## Standard Transformers generation

```python
prompt = model.build_grounded_prompt(
    "Create a defensive credential-protection exercise.",
    language="en",
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=384)
completion = outputs[:, inputs["input_ids"].shape[1]:]
print(tokenizer.decode(completion[0], skip_special_tokens=True))
```

## Safety and intended use

This model is intended for authorized defensive training, awareness, tabletop
exercises, architecture reviews, and incident-response preparation. Outputs should be
reviewed by qualified cybersecurity professionals. It is not intended to provide
operational exploitation instructions or to support unauthorized access.

## Custom code

This repository contains a custom Transformers architecture. Use
`trust_remote_code=True` only after reviewing and trusting the repository code.
'''
    (output_dir / "README.md").write_text(card, encoding="utf-8")


def copy_remote_code(output_dir: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "src" / "asg_transformer" / "hf"
    for filename in ("configuration_asg_transformer.py", "modeling_asg_transformer.py"):
        shutil.copy2(source / filename, output_dir / filename)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    knowledge_dir = Path(args.knowledge_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        revision=args.revision,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        revision=args.revision,
        torch_dtype=resolve_dtype(args.torch_dtype),
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True,
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            base_model.resize_token_embeddings(len(tokenizer))
        else:
            tokenizer.pad_token = tokenizer.eos_token

    config = ASGTransformerConfig(
        base_model_config=base_model.config.to_dict(),
        semantic_projection_dim=args.semantic_dim,
        duration_bins=parse_duration_bins(args.duration_bins),
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        vocab_size=len(tokenizer),
        torch_dtype=str(base_model.dtype).replace("torch.", ""),
        name_or_path=args.base_model,
    )
    model = ASGTransformerForCausalLM(config)
    load_result = model.generator.load_state_dict(base_model.state_dict(), strict=False)
    if load_result.unexpected_keys:
        raise RuntimeError(f"Unexpected base-model tensors: {load_result.unexpected_keys[:20]}")
    if load_result.missing_keys:
        print(
            "Warning: generator parameters not initialized from the base model: "
            f"{len(load_result.missing_keys)}"
        )

    model.save_pretrained(
        output_dir,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    tokenizer.save_pretrained(output_dir)
    model.generation_config.save_pretrained(output_dir)
    copy_remote_code(output_dir)
    copy_knowledge(knowledge_dir, output_dir)
    write_model_card(output_dir, args.base_model)

    if args.repo_id:
        api = HfApi()
        api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            repo_id=args.repo_id,
            repo_type="model",
            folder_path=str(output_dir),
            commit_message="Publish ASGTransformer checkpoint",
        )
        print(f"Published model: https://huggingface.co/{args.repo_id}")

    print(f"ASGTransformer repository created at: {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from transformers import AutoModelForCausalLM, AutoTokenizer

from asg_transformer.hf import ASGTransformerConfig, ASGTransformerForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ASGTransformer as one Hugging Face AutoModelForCausalLM repository."
    )
    parser.add_argument("--base-model", required=True, help="Base causal language model ID or path")
    parser.add_argument("--output-dir", default="dist/ASGTransformer")
    parser.add_argument("--knowledge-dir", default="data/processed")
    parser.add_argument("--semantic-dim", type=int, default=256)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def copy_knowledge(source_dir: Path, output_dir: Path) -> None:
    target = output_dir / "knowledge"
    target.mkdir(parents=True, exist_ok=True)
    catalog: dict[str, object] = {}
    if source_dir.is_dir():
        for source in sorted(source_dir.glob("*.json")):
            shutil.copy2(source, target / source.name)
            try:
                catalog[source.stem] = json.loads(source.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
    (target / "catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    template = """You are ASGTransformer, a professional defensive cybersecurity scenario generator.
Use the supplied task and packaged knowledge to produce a clear, authorized, high-level scenario.
Include: title, executive summary, ordered stages, estimated duration, defensive observations,
and expected outcomes. Do not provide operational exploitation instructions.

User task:
{input}

Professional response:
"""
    (target / "prompt_template.txt").write_text(template, encoding="utf-8")


def write_model_card(output_dir: Path, base_model: str) -> None:
    card = f'''---
library_name: transformers
pipeline_tag: text-generation
tags:
- custom-code
- cybersecurity
- defensive-security
- causal-lm
- scenario-generation
language:
- en
- ar
license: mit
base_model: {base_model}
---

# ASGTransformer

A unified Hugging Face Transformers model containing a causal text generator,
semantic encoder, duration prediction head, and scenario planning head in one
`model.safetensors` checkpoint.

## Load

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("asgmodel/ASGTransformer", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "asgmodel/ASGTransformer",
    trust_remote_code=True,
)

prompt = "Create an authorized defensive scenario for phishing awareness."
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

Custom Hub code should only be loaded from repositories you trust and have reviewed.
'''
    (output_dir / "README.md").write_text(card, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model)
    base_config = base_model.config.to_dict()

    config = ASGTransformerConfig(
        base_model_config=base_config,
        semantic_projection_dim=args.semantic_dim,
        bos_token_id=base_model.config.bos_token_id,
        eos_token_id=base_model.config.eos_token_id,
        pad_token_id=base_model.config.pad_token_id,
        tie_word_embeddings=getattr(base_model.config, "tie_word_embeddings", True),
        architectures=["ASGTransformerForCausalLM"],
        auto_map={
            "AutoConfig": "configuration_asg_transformer.ASGTransformerConfig",
            "AutoModelForCausalLM": "modeling_asg_transformer.ASGTransformerForCausalLM",
        },
    )
    model = ASGTransformerForCausalLM(config)
    missing, unexpected = model.generator.load_state_dict(base_model.state_dict(), strict=False)
    if unexpected:
        raise RuntimeError(f"Unexpected base-model weights: {unexpected}")
    if missing:
        print(f"Warning: {len(missing)} generator parameters were not initialized from base model")

    ASGTransformerConfig.register_for_auto_class("AutoConfig")
    ASGTransformerForCausalLM.register_for_auto_class("AutoModelForCausalLM")
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)
    model.generation_config.save_pretrained(output_dir)

    source_root = Path(__file__).resolve().parents[1] / "src" / "asg_transformer" / "hf"
    shutil.copy2(source_root / "configuration_asg_transformer.py", output_dir / "configuration_asg_transformer.py")
    shutil.copy2(source_root / "modeling_asg_transformer.py", output_dir / "modeling_asg_transformer.py")
    copy_knowledge(Path(args.knowledge_dir), output_dir)
    write_model_card(output_dir, args.base_model)

    if args.repo_id:
        model.push_to_hub(args.repo_id, private=args.private, safe_serialization=True)
        tokenizer.push_to_hub(args.repo_id, private=args.private)
        from huggingface_hub import HfApi
        HfApi().upload_folder(repo_id=args.repo_id, folder_path=str(output_dir), repo_type="model")

    print(f"ASGTransformer exported to: {output_dir}")


if __name__ == "__main__":
    main()

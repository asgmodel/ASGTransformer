#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

REQUIRED_PREFIXES = (
    "generator.",
    "semantic_encoder.",
    "duration_head.",
    "scenario_head.",
)
REQUIRED_FILES = (
    "config.json",
    "tokenizer_config.json",
    "configuration_asg_transformer.py",
    "modeling_asg_transformer.py",
    "README.md",
    "knowledge/catalog.json",
    "knowledge/prompt_template.txt",
)


def checkpoint_files(model_dir: Path) -> list[Path]:
    index_file = model_dir / "model.safetensors.index.json"
    if index_file.is_file():
        index = json.loads(index_file.read_text(encoding="utf-8"))
        return sorted({model_dir / filename for filename in index["weight_map"].values()})
    single = model_dir / "model.safetensors"
    return [single] if single.is_file() else []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an exported ASGTransformer Hugging Face repository"
    )
    parser.add_argument("model_dir")
    parser.add_argument("--skip-generation", action="store_true")
    args = parser.parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()

    missing_files = [name for name in REQUIRED_FILES if not (model_dir / name).is_file()]
    if missing_files:
        raise RuntimeError(f"Repository is missing required files: {missing_files}")

    files = checkpoint_files(model_dir)
    if not files or any(not file.is_file() for file in files):
        raise FileNotFoundError("No complete Safetensors checkpoint was found")

    tensor_keys: set[str] = set()
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as checkpoint:
            tensor_keys.update(checkpoint.keys())

    missing_groups = [
        prefix for prefix in REQUIRED_PREFIXES if not any(key.startswith(prefix) for key in tensor_keys)
    ]
    if missing_groups:
        raise RuntimeError(f"Checkpoint is missing parameter groups: {missing_groups}")

    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).eval()

    if config.model_type != "asg_transformer":
        raise RuntimeError(f"Unexpected model_type: {config.model_type}")
    if model.__class__.__name__ != "ASGTransformerForCausalLM":
        raise RuntimeError(f"Unexpected model class: {model.__class__.__name__}")

    sample = tokenizer("Create a defensive training scenario.", return_tensors="pt")
    with torch.inference_mode():
        outputs = model(**sample)
    if outputs.logits.shape[:2] != sample["input_ids"].shape:
        raise RuntimeError("Forward-pass logits have an invalid shape")
    if outputs.semantic_embedding.shape[-1] != config.semantic_projection_dim:
        raise RuntimeError("Semantic embedding dimension does not match config")

    if not args.skip_generation:
        with torch.inference_mode():
            generated = model.generate(**sample, max_new_tokens=2, do_sample=False)
        if generated.shape[1] <= sample["input_ids"].shape[1]:
            raise RuntimeError("Generation did not produce new tokens")

    print(f"Verified repository: {model_dir}")
    print(f"Safetensors files: {len(files)}")
    print(f"Tensor count: {len(tensor_keys)}")
    print("AutoConfig, AutoTokenizer, AutoModelForCausalLM, forward, and generation: OK")


if __name__ == "__main__":
    main()

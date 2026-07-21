#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

REQUIRED_PREFIXES = (
    "generator.",
    "semantic_encoder.",
    "duration_head.",
    "scenario_head.",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify an exported ASGTransformer checkpoint")
    parser.add_argument("model_dir")
    args = parser.parse_args()
    model_dir = Path(args.model_dir).resolve()

    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        files = sorted(model_dir.glob("model-*.safetensors"))
    if not files:
        raise FileNotFoundError("No safetensors checkpoint found")

    keys: set[str] = set()
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as checkpoint:
            keys.update(checkpoint.keys())

    missing = [prefix for prefix in REQUIRED_PREFIXES if not any(k.startswith(prefix) for k in keys)]
    if missing:
        raise RuntimeError(f"Checkpoint is missing required parameter groups: {missing}")

    AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    print(f"Verified {len(keys)} tensors across {len(files)} checkpoint file(s).")
    print("AutoTokenizer and AutoModelForCausalLM loading succeeded.")


if __name__ == "__main__":
    main()

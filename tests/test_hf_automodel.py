from __future__ import annotations

import json
from pathlib import Path
import shutil

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast

from asg_transformer.hf import ASGTransformerConfig, ASGTransformerForCausalLM


def make_tokenizer(path: Path) -> PreTrainedTokenizerFast:
    vocabulary = {
        "<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
        "Create": 4, "a": 5, "defensive": 6, "scenario": 7,
        "for": 8, "training": 9, ".": 10,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>", bos_token="<bos>", eos_token="<eos>", unk_token="<unk>",
    )
    tokenizer.save_pretrained(path)
    return tokenizer


def test_save_and_load_with_auto_model(tmp_path: Path) -> None:
    config = ASGTransformerConfig(
        base_model_config={
            "model_type": "gpt2", "vocab_size": 11, "n_positions": 32,
            "n_ctx": 32, "n_embd": 16, "n_layer": 1, "n_head": 2,
            "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
        },
        semantic_projection_dim=8,
        duration_bins=[5, 10, 20],
        num_scenario_labels=4,
        bos_token_id=1, eos_token_id=2, pad_token_id=0,
        architectures=["ASGTransformerForCausalLM"],
        auto_map={
            "AutoConfig": "configuration_asg_transformer.ASGTransformerConfig",
            "AutoModelForCausalLM": "modeling_asg_transformer.ASGTransformerForCausalLM",
        },
    )
    model = ASGTransformerForCausalLM(config).eval()
    make_tokenizer(tmp_path)
    model.save_pretrained(tmp_path, safe_serialization=True)

    source = Path(__file__).parents[1] / "src" / "asg_transformer" / "hf"
    shutil.copy2(source / "configuration_asg_transformer.py", tmp_path / "configuration_asg_transformer.py")
    shutil.copy2(source / "modeling_asg_transformer.py", tmp_path / "modeling_asg_transformer.py")
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "knowledge" / "catalog.json").write_text("{}", encoding="utf-8")
    (tmp_path / "knowledge" / "prompt_template.txt").write_text("Task: {input}\nResponse:", encoding="utf-8")

    tokenizer = AutoTokenizer.from_pretrained(tmp_path, trust_remote_code=True)
    loaded = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True).eval()
    batch = tokenizer("Create a defensive scenario .", return_tensors="pt")
    with torch.no_grad():
        result = loaded(**batch)
    assert result.logits.shape[:2] == batch["input_ids"].shape
    assert result.semantic_embedding.shape[-1] == 8
    assert result.duration_logits.shape[-1] == 3
    assert result.scenario_logits.shape[-1] == 4
    generated = loaded.generate(**batch, max_new_tokens=2, do_sample=False)
    sequences = generated.sequences if hasattr(generated, "sequences") else generated
    assert sequences.shape[1] >= batch["input_ids"].shape[1]

    checkpoint = tmp_path / "model.safetensors"
    assert checkpoint.is_file()
    index = json.loads((tmp_path / "config.json").read_text())
    assert index["model_type"] == "asg_transformer"
    assert "AutoModelForCausalLM" in index["auto_map"]


def test_all_heads_roundtrip(tmp_path: Path) -> None:
    config = ASGTransformerConfig(
        base_model_config={
            "model_type": "gpt2", "vocab_size": 13, "n_positions": 16,
            "n_ctx": 16, "n_embd": 12, "n_layer": 1, "n_head": 2,
            "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
        },
        semantic_projection_dim=6, duration_bins=[5, 15], num_scenario_labels=3,
    )
    model = ASGTransformerForCausalLM(config)
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.save_pretrained(tmp_path, safe_serialization=True)
    loaded = ASGTransformerForCausalLM.from_pretrained(tmp_path)
    after = loaded.state_dict()
    assert before.keys() == after.keys()
    for key in before:
        assert torch.equal(before[key], after[key]), key

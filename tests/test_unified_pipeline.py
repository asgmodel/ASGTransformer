from __future__ import annotations

import json
from pathlib import Path

from asg_transformer import ASGTransformer, ASGTransformerConfig
from asg_transformer.hf import ASGTransformerForCausalLM


def tiny_config() -> ASGTransformerConfig:
    return ASGTransformerConfig(
        base_model_config={
            "model_type": "gpt2",
            "vocab_size": 32,
            "n_positions": 32,
            "n_ctx": 32,
            "n_embd": 16,
            "n_layer": 1,
            "n_head": 2,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": 0,
        },
        semantic_projection_dim=8,
        duration_bins=[5, 15, 30],
        scenario_labels=["awareness", "response"],
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


def test_public_alias_points_to_hf_model() -> None:
    assert ASGTransformer is ASGTransformerForCausalLM


def test_packaged_knowledge_prompt(tmp_path: Path) -> None:
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    (knowledge_dir / "catalog.json").write_text(
        json.dumps(
            {
                "techniques": [
                    {
                        "label": "Phishing Awareness",
                        "description": "Defensive email awareness exercise",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (knowledge_dir / "prompt_template.txt").write_text(
        "Language: {language}\nTask: {input}\nKnowledge: {knowledge}\nResponse:",
        encoding="utf-8",
    )

    model = ASGTransformer(tiny_config())
    prompt = model.build_grounded_prompt(
        "Create a phishing awareness exercise",
        model_dir=tmp_path,
        language="en",
    )
    assert "Phishing Awareness" in prompt
    assert "Create a phishing awareness exercise" in prompt

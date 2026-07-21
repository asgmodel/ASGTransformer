from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from asg_transformer.config import settings


class ASGTransformerService:
    """Application service backed by one Hugging Face ASGTransformer repository."""

    def __init__(self) -> None:
        source = settings.model_source
        self.tokenizer = AutoTokenizer.from_pretrained(
            source,
            trust_remote_code=settings.trust_remote_code,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            source,
            trust_remote_code=settings.trust_remote_code,
            torch_dtype=settings.torch_dtype,
            device_map=settings.device_map,
        ).eval()
        self.source = source

    def generate_scenario(self, text: str, **kwargs: Any) -> dict[str, Any]:
        language = kwargs.pop("language", settings.default_language)
        max_new_tokens = kwargs.pop("max_new_tokens", settings.max_new_tokens)
        result = self.model.generate_scenario(
            self.tokenizer,
            text,
            language=language,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )
        result["knowledge_matches"] = self.model.retrieve_knowledge(text)
        return result

    @torch.inference_mode()
    def encode(self, text: str) -> list[float]:
        batch = self.tokenizer(text, return_tensors="pt")
        device = next(self.model.parameters()).device
        batch = {key: value.to(device) for key, value in batch.items()}
        embedding = self.model.encode(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )
        return embedding[0].cpu().tolist()


@lru_cache(maxsize=1)
def get_service() -> ASGTransformerService:
    return ASGTransformerService()

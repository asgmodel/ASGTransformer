from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput
from transformers.utils.hub import cached_file

from .configuration_asg_transformer import ASGTransformerConfig


@dataclass
class ASGTransformerCausalLMOutput(ModelOutput):
    """Causal-LM output enriched with ASGTransformer auxiliary predictions."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Any | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    semantic_embedding: torch.FloatTensor | None = None
    duration_logits: torch.FloatTensor | None = None
    scenario_logits: torch.FloatTensor | None = None


class ASGTransformerPreTrainedModel(PreTrainedModel):
    """Shared Hugging Face base class for ASGTransformer checkpoints."""

    config_class = ASGTransformerConfig
    base_model_prefix = "asg_transformer"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ASGTransformerForCausalLM"]

    def _init_weights(self, module: nn.Module) -> None:
        initializer_range = float(getattr(self.config, "initializer_range", 0.02))
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class ASGTransformerForCausalLM(ASGTransformerPreTrainedModel, GenerationMixin):
    """Unified ASG text-to-text model compatible with ``AutoModelForCausalLM``.

    The complete base causal language model and every ASG-specific trainable
    component are registered beneath this module. Consequently, a single call
    to :meth:`save_pretrained` stores the generator, semantic encoder, duration
    head, and scenario head in the same Safetensors checkpoint (or checkpoint
    shard set when the selected base model is large).
    """

    _tied_weights_keys: list[str] = []

    def __init__(self, config: ASGTransformerConfig) -> None:
        super().__init__(config)
        base_config = AutoConfig.for_model(
            config.base_model_config["model_type"],
            **{
                key: value
                for key, value in config.base_model_config.items()
                if key != "model_type"
            },
        )
        base_config.output_hidden_states = True
        # Explicitly persist input and output embeddings in the ASG checkpoint.
        base_config.tie_word_embeddings = False

        self.generator = AutoModelForCausalLM.from_config(base_config)
        hidden_size = self._resolve_hidden_size(base_config)

        self.semantic_encoder = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, config.semantic_projection_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.semantic_projection_dim, config.semantic_projection_dim),
        )
        self.duration_head = nn.Sequential(
            nn.LayerNorm(config.semantic_projection_dim),
            nn.Linear(config.semantic_projection_dim, len(config.duration_bins)),
        )
        self.scenario_head = nn.Sequential(
            nn.LayerNorm(config.semantic_projection_dim),
            nn.Linear(config.semantic_projection_dim, config.num_scenario_labels),
        )
        self.post_init()

    @staticmethod
    def _resolve_hidden_size(base_config: Any) -> int:
        for name in ("hidden_size", "n_embd", "d_model", "dim"):
            value = getattr(base_config, name, None)
            if value is not None:
                return int(value)
        raise ValueError("Unable to determine the base model hidden size")

    @property
    def base_model(self) -> nn.Module:
        return self.generator

    def get_input_embeddings(self) -> nn.Module:
        return self.generator.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.generator.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.generator.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.generator.set_output_embeddings(new_embeddings)

    def tie_weights(self, *args: Any, **kwargs: Any) -> None:
        # The exported checkpoint intentionally keeps explicit input/output tensors.
        return None

    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = None,
        **kwargs: Any,
    ) -> nn.Module:
        embeddings = self.generator.resize_token_embeddings(new_num_tokens, **kwargs)
        if new_num_tokens is not None:
            self.config.base_model_config["vocab_size"] = int(new_num_tokens)
            self.config.vocab_size = int(new_num_tokens)
        return embeddings

    @staticmethod
    def masked_mean(
        hidden: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden.mean(dim=1)
        mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denominator

    def encode(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return normalized semantic representations for input sequences."""
        outputs = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        pooled = self.masked_mean(outputs.hidden_states[-1], attention_mask)
        return F.normalize(self.semantic_encoder(pooled), dim=-1)

    def predict_duration(
        self,
        semantic_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict a duration class and map it to a minute value."""
        logits = self.duration_head(semantic_embedding)
        indices = logits.argmax(dim=-1)
        bins = torch.as_tensor(
            self.config.duration_bins,
            device=logits.device,
            dtype=torch.long,
        )
        return logits, bins[indices]

    def predict_scenario_type(
        self,
        semantic_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, list[str]]:
        """Predict one configured high-level scenario category per sequence."""
        logits = self.scenario_head(semantic_embedding)
        indices = logits.argmax(dim=-1).tolist()
        labels = [self.config.scenario_labels[int(index)] for index in indices]
        return logits, labels

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        duration_labels: torch.LongTensor | None = None,
        scenario_labels: torch.LongTensor | None = None,
        semantic_targets: torch.FloatTensor | None = None,
        past_key_values: Any | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> ASGTransformerCausalLMOutput | tuple[Any, ...]:
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        outputs: CausalLMOutputWithPast = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )

        hidden = outputs.hidden_states[-1]
        pooled = self.masked_mean(hidden, attention_mask)
        semantic_embedding = F.normalize(self.semantic_encoder(pooled), dim=-1)
        duration_logits = self.duration_head(semantic_embedding)
        scenario_logits = self.scenario_head(semantic_embedding)

        auxiliary_loss = hidden.new_zeros(())
        if semantic_targets is not None:
            if semantic_targets.shape != semantic_embedding.shape:
                raise ValueError(
                    "semantic_targets must have shape "
                    f"{tuple(semantic_embedding.shape)}, received {tuple(semantic_targets.shape)}"
                )
            targets = F.normalize(semantic_targets, dim=-1)
            semantic_loss = 1.0 - F.cosine_similarity(
                semantic_embedding,
                targets,
                dim=-1,
            ).mean()
            auxiliary_loss = auxiliary_loss + self.config.semantic_loss_weight * semantic_loss
        if duration_labels is not None:
            auxiliary_loss = auxiliary_loss + self.config.duration_loss_weight * F.cross_entropy(
                duration_logits,
                duration_labels,
            )
        if scenario_labels is not None:
            auxiliary_loss = auxiliary_loss + self.config.scenario_loss_weight * F.cross_entropy(
                scenario_logits,
                scenario_labels,
            )

        loss = outputs.loss
        has_auxiliary_targets = any(
            item is not None
            for item in (semantic_targets, duration_labels, scenario_labels)
        )
        if has_auxiliary_targets:
            loss = auxiliary_loss if loss is None else loss + auxiliary_loss

        if not return_dict:
            return (
                loss,
                outputs.logits,
                outputs.past_key_values,
                semantic_embedding,
                duration_logits,
                scenario_logits,
            )

        return ASGTransformerCausalLMOutput(
            loss=loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states if output_hidden_states else None,
            attentions=outputs.attentions,
            semantic_embedding=semantic_embedding,
            duration_logits=duration_logits,
            scenario_logits=scenario_logits,
        )

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Run standard Hugging Face generation through the bundled causal LM."""
        return self.generator.generate(*args, **kwargs)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        **kwargs: Any,
    ) -> dict[str, Any]:
        return self.generator.prepare_inputs_for_generation(input_ids, **kwargs)

    def _reorder_cache(
        self,
        past_key_values: Any,
        beam_idx: torch.LongTensor,
    ) -> Any:
        reorder = getattr(self.generator, "_reorder_cache", None)
        return reorder(past_key_values, beam_idx) if reorder is not None else past_key_values

    @staticmethod
    def _tokenize_for_retrieval(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[\w\-]+", text.casefold(), flags=re.UNICODE)
            if len(token) > 2
        }

    @staticmethod
    def _iter_catalog_records(value: Any, path: str = "") -> Iterable[dict[str, str]]:
        if isinstance(value, dict):
            # Treat dictionaries containing descriptive fields as records.
            descriptive_keys = {
                "id",
                "name",
                "label",
                "title",
                "description",
                "tactic",
                "technique",
            }
            if descriptive_keys.intersection(value):
                text = " | ".join(
                    f"{key}: {item}"
                    for key, item in value.items()
                    if isinstance(item, (str, int, float, bool)) and str(item).strip()
                )
                if text:
                    yield {"source": path or "catalog", "text": text}
            for key, item in value.items():
                child = f"{path}.{key}" if path else str(key)
                yield from ASGTransformerForCausalLM._iter_catalog_records(item, child)
        elif isinstance(value, list):
            for index, item in enumerate(value):
                yield from ASGTransformerForCausalLM._iter_catalog_records(
                    item,
                    f"{path}[{index}]",
                )
        elif isinstance(value, str) and value.strip():
            yield {"source": path or "catalog", "text": value.strip()}

    def _resolve_asset(self, relative_path: str, model_dir: str | Path | None = None) -> Path | None:
        if model_dir is not None:
            candidate = Path(model_dir).expanduser() / relative_path
            if candidate.is_file():
                return candidate.resolve()

        source = getattr(self.config, "_name_or_path", None)
        if not source:
            return None
        local_candidate = Path(str(source)).expanduser() / relative_path
        if local_candidate.is_file():
            return local_candidate.resolve()

        try:
            resolved = cached_file(str(source), relative_path, _raise_exceptions_for_missing_entries=False)
        except (OSError, ValueError):
            resolved = None
        return Path(resolved) if resolved else None

    def load_knowledge(self, model_dir: str | Path | None = None) -> dict[str, Any]:
        """Load the packaged knowledge catalog from a local or Hub checkpoint."""
        path = self._resolve_asset(self.config.knowledge_file, model_dir)
        if path is None:
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Unable to read packaged knowledge file: {path}") from exc
        if not isinstance(value, dict):
            raise ValueError("The packaged knowledge catalog must contain a JSON object")
        return value

    def retrieve_knowledge(
        self,
        text: str,
        model_dir: str | Path | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return lightweight lexical matches from the packaged catalog.

        This deterministic retrieval layer keeps prompt construction self-contained
        and requires no second model. Fine-tuned semantic retrieval remains available
        through :meth:`encode` for application-level vector indexes.
        """
        query_tokens = self._tokenize_for_retrieval(text)
        if not query_tokens:
            return []
        records = list(self._iter_catalog_records(self.load_knowledge(model_dir)))
        ranked: list[dict[str, Any]] = []
        for record in records:
            record_tokens = self._tokenize_for_retrieval(record["text"])
            overlap = query_tokens.intersection(record_tokens)
            if not overlap:
                continue
            score = len(overlap) / max(len(query_tokens), 1)
            ranked.append({**record, "score": round(score, 6)})
        ranked.sort(key=lambda item: (item["score"], item["source"]), reverse=True)
        limit = self.config.knowledge_top_k if top_k is None else max(int(top_k), 0)
        return ranked[:limit]

    def build_grounded_prompt(
        self,
        text: str,
        model_dir: str | Path | None = None,
        language: str = "en",
        top_k: int | None = None,
    ) -> str:
        """Build a professional prompt augmented with packaged knowledge matches."""
        clean_text = text.strip()
        if not clean_text:
            raise ValueError("text must not be empty")

        template = (
            "You are ASGTransformer, a professional defensive cybersecurity "
            "scenario-generation model.\n"
            "Create only authorized, high-level training content. Do not provide "
            "exploit code, payloads, evasion procedures, credential theft steps, "
            "or instructions that enable real-world compromise.\n\n"
            "Response language: {language}\n"
            "User task:\n{input}\n\n"
            "Grounding context:\n{knowledge}\n\n"
            "Required response:\n"
            "1. Scenario title\n"
            "2. Executive summary\n"
            "3. Assumptions and scope\n"
            "4. Ordered defensive stages\n"
            "5. Estimated duration\n"
            "6. Monitoring and detection objectives\n"
            "7. Expected learning outcomes\n\n"
            "Professional response:\n"
        )
        template_path = self._resolve_asset(self.config.prompt_template_file, model_dir)
        if template_path is not None:
            template = template_path.read_text(encoding="utf-8")

        matches = self.retrieve_knowledge(clean_text, model_dir=model_dir, top_k=top_k)
        if matches:
            knowledge = "\n".join(
                f"- [{item['source']}] {item['text']}"
                for item in matches
            )[: self.config.max_knowledge_chars]
        else:
            knowledge = "No matching packaged records were found; remain within the user-provided scope."

        return template.format(
            input=clean_text,
            knowledge=knowledge,
            language=language,
        )

    @torch.inference_mode()
    def generate_scenario(
        self,
        tokenizer: Any,
        text: str,
        *,
        model_dir: str | Path | None = None,
        language: str = "en",
        max_new_tokens: int = 384,
        **generation_kwargs: Any,
    ) -> dict[str, Any]:
        """Convenience text-to-text API returning text and auxiliary predictions."""
        prompt = self.build_grounded_prompt(
            text,
            model_dir=model_dir,
            language=language,
        )
        device = next(self.parameters()).device
        batch = tokenizer(prompt, return_tensors="pt")
        batch = {key: value.to(device) for key, value in batch.items()}

        semantic_embedding = self.encode(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
        )
        duration_logits, duration_minutes = self.predict_duration(semantic_embedding)
        scenario_logits, scenario_types = self.predict_scenario_type(semantic_embedding)

        defaults = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        }
        defaults.update(generation_kwargs)
        generated = self.generate(**batch, **defaults)
        sequences = generated.sequences if hasattr(generated, "sequences") else generated
        prompt_length = batch["input_ids"].shape[1]
        completion_ids = sequences[:, prompt_length:]
        generated_text = tokenizer.decode(completion_ids[0], skip_special_tokens=True).strip()

        return {
            "text": generated_text,
            "estimated_duration_minutes": int(duration_minutes[0].item()),
            "duration_probabilities": torch.softmax(duration_logits[0], dim=-1).cpu().tolist(),
            "scenario_type": scenario_types[0],
            "scenario_probabilities": torch.softmax(scenario_logits[0], dim=-1).cpu().tolist(),
            "semantic_embedding": semantic_embedding[0].cpu().tolist(),
            "prompt": prompt,
        }

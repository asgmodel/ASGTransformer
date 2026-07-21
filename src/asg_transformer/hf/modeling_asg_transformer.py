from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import ModelOutput

from .configuration_asg_transformer import ASGTransformerConfig


@dataclass
class ASGTransformerCausalLMOutput(ModelOutput):
    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None
    past_key_values: Any | None = None
    hidden_states: tuple[torch.FloatTensor, ...] | None = None
    attentions: tuple[torch.FloatTensor, ...] | None = None
    semantic_embedding: torch.FloatTensor | None = None
    duration_logits: torch.FloatTensor | None = None
    scenario_logits: torch.FloatTensor | None = None


class ASGTransformerPreTrainedModel(PreTrainedModel):
    config_class = ASGTransformerConfig
    base_model_prefix = "asg"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ASGTransformerForCausalLM"]

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range or 0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class ASGTransformerForCausalLM(ASGTransformerPreTrainedModel, GenerationMixin):
    """Unified text-to-text ASG model.

    Pipeline inside one ``PreTrainedModel``:
      hidden states -> semantic encoder -> duration/scenario planners -> LM generator.

    All trainable parameters are registered PyTorch submodules and therefore
    saved into the same ``model.safetensors`` checkpoint by ``save_pretrained``.
    """

    _tied_weights_keys = []

    def __init__(self, config: ASGTransformerConfig) -> None:
        super().__init__(config)
        base_config = AutoConfig.for_model(
            config.base_model_config["model_type"],
            **{k: v for k, v in config.base_model_config.items() if k != "model_type"},
        )
        base_config.output_hidden_states = True
        # Store input and output embeddings as explicit tensors in one checkpoint.
        base_config.tie_word_embeddings = False
        self.generator = AutoModelForCausalLM.from_config(base_config)

        hidden_size = self._resolve_hidden_size(base_config)
        self.semantic_encoder = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, config.semantic_projection_dim),
            nn.GELU(),
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
        for name in ("hidden_size", "n_embd", "d_model"):
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
        # ASGTransformer deliberately stores untied input/output embeddings so
        # every tensor is explicit in the single safetensors checkpoint.
        return None

    def resize_token_embeddings(self, new_num_tokens: int | None = None, **kwargs: Any):
        embeddings = self.generator.resize_token_embeddings(new_num_tokens, **kwargs)
        if new_num_tokens is not None:
            self.config.base_model_config["vocab_size"] = int(new_num_tokens)
        return embeddings

    @staticmethod
    def masked_mean(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        if attention_mask is None:
            return hidden.mean(dim=1)
        mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return (hidden * mask).sum(dim=1) / denominator

    def encode(self, input_ids: torch.LongTensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        outputs = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        pooled = self.masked_mean(outputs.hidden_states[-1], attention_mask)
        return F.normalize(self.semantic_encoder(pooled), dim=-1)

    def predict_duration(self, semantic_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.duration_head(semantic_embedding)
        indices = logits.argmax(dim=-1)
        bins = torch.tensor(self.config.duration_bins, device=logits.device)
        return logits, bins[indices]

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
    ) -> ASGTransformerCausalLMOutput | tuple:
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

        loss = outputs.loss
        auxiliary_loss = torch.zeros((), device=hidden.device, dtype=hidden.dtype)
        if semantic_targets is not None:
            auxiliary_loss = auxiliary_loss + self.config.semantic_loss_weight * (
                1.0 - F.cosine_similarity(semantic_embedding, semantic_targets, dim=-1).mean()
            )
        if duration_labels is not None:
            auxiliary_loss = auxiliary_loss + self.config.duration_loss_weight * F.cross_entropy(
                duration_logits, duration_labels
            )
        if scenario_labels is not None:
            auxiliary_loss = auxiliary_loss + self.config.scenario_loss_weight * F.cross_entropy(
                scenario_logits, scenario_labels
            )
        loss = auxiliary_loss if loss is None else loss + auxiliary_loss

        if not return_dict:
            return (loss, outputs.logits, outputs.past_key_values, semantic_embedding, duration_logits, scenario_logits)
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

    def generate(self, *args: Any, **kwargs: Any):
        """Delegate standard Hugging Face generation to the bundled causal LM."""
        return self.generator.generate(*args, **kwargs)

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs: Any) -> dict[str, Any]:
        prepared = self.generator.prepare_inputs_for_generation(input_ids, **kwargs)
        return prepared

    def _reorder_cache(self, past_key_values: Any, beam_idx: torch.LongTensor) -> Any:
        reorder = getattr(self.generator, "_reorder_cache", None)
        return reorder(past_key_values, beam_idx) if reorder is not None else past_key_values

    def build_grounded_prompt(self, text: str, model_dir: str | Path | None = None) -> str:
        """Build a safe, knowledge-grounded generation prompt from packaged assets."""
        template = (
            "You are ASGTransformer, a defensive cybersecurity scenario assistant.\n"
            "Create an authorized, high-level training scenario. Avoid operational exploitation instructions.\n"
            "User task:\n{input}\n\nProfessional response:\n"
        )
        if model_dir is not None:
            path = Path(model_dir) / self.config.prompt_template_file
            if path.is_file():
                template = path.read_text(encoding="utf-8")
        return template.format(input=text.strip())

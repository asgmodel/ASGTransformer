from __future__ import annotations

from typing import Any
from transformers import PretrainedConfig


class ASGTransformerConfig(PretrainedConfig):
    """Configuration for :class:`ASGTransformerForCausalLM`.

    The configuration embeds the complete base language-model configuration and
    the ASG-specific semantic, duration, and planning head settings.
    """

    model_type = "asg_transformer"

    def __init__(
        self,
        base_model_config: dict[str, Any] | None = None,
        semantic_projection_dim: int = 256,
        initializer_range: float = 0.02,
        duration_bins: list[int] | None = None,
        num_scenario_labels: int = 8,
        semantic_loss_weight: float = 0.10,
        duration_loss_weight: float = 0.10,
        scenario_loss_weight: float = 0.10,
        knowledge_file: str = "knowledge/catalog.json",
        prompt_template_file: str = "knowledge/prompt_template.txt",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if base_model_config is None:
            base_model_config = {
                "model_type": "gpt2",
                "vocab_size": 50257,
                "n_positions": 1024,
                "n_embd": 768,
                "n_layer": 12,
                "n_head": 12,
            }
        self.base_model_config = dict(base_model_config)
        self.semantic_projection_dim = int(semantic_projection_dim)
        self.initializer_range = float(initializer_range)
        self.duration_bins = list(duration_bins or [5, 10, 15, 30, 45, 60, 90, 120])
        self.num_scenario_labels = int(num_scenario_labels)
        self.semantic_loss_weight = float(semantic_loss_weight)
        self.duration_loss_weight = float(duration_loss_weight)
        self.scenario_loss_weight = float(scenario_loss_weight)
        self.knowledge_file = str(knowledge_file)
        self.prompt_template_file = str(prompt_template_file)

        if self.semantic_projection_dim < 1:
            raise ValueError("semantic_projection_dim must be positive")
        if not self.duration_bins:
            raise ValueError("duration_bins must not be empty")
        if self.num_scenario_labels < 1:
            raise ValueError("num_scenario_labels must be positive")

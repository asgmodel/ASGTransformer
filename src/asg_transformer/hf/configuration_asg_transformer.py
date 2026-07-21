from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


class ASGTransformerConfig(PretrainedConfig):
    """Configuration for :class:`ASGTransformerForCausalLM`.

    The configuration stores the complete causal language-model configuration
    together with ASGTransformer's semantic representation, duration estimation,
    scenario classification, and packaged-knowledge settings.
    """

    model_type = "asg_transformer"

    def __init__(
        self,
        base_model_config: dict[str, Any] | None = None,
        semantic_projection_dim: int = 256,
        initializer_range: float = 0.02,
        duration_bins: list[int] | None = None,
        scenario_labels: list[str] | None = None,
        semantic_loss_weight: float = 0.10,
        duration_loss_weight: float = 0.10,
        scenario_loss_weight: float = 0.10,
        knowledge_file: str = "knowledge/catalog.json",
        prompt_template_file: str = "knowledge/prompt_template.txt",
        knowledge_top_k: int = 8,
        max_knowledge_chars: int = 6_000,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("architectures", ["ASGTransformerForCausalLM"])
        kwargs.setdefault(
            "auto_map",
            {
                "AutoConfig": "configuration_asg_transformer.ASGTransformerConfig",
                "AutoModelForCausalLM": (
                    "modeling_asg_transformer.ASGTransformerForCausalLM"
                ),
            },
        )
        super().__init__(**kwargs)

        if base_model_config is None:
            base_model_config = {
                "model_type": "gpt2",
                "vocab_size": 50_257,
                "n_positions": 1_024,
                "n_ctx": 1_024,
                "n_embd": 768,
                "n_layer": 12,
                "n_head": 12,
            }

        self.base_model_config = dict(base_model_config)
        self.semantic_projection_dim = int(semantic_projection_dim)
        self.initializer_range = float(initializer_range)
        self.duration_bins = [
            int(value) for value in (duration_bins or [5, 10, 15, 30, 45, 60, 90, 120])
        ]
        self.scenario_labels = list(
            scenario_labels
            or [
                "awareness",
                "initial_access",
                "credential_protection",
                "lateral_movement_detection",
                "persistence_detection",
                "incident_response",
                "recovery",
                "executive_tabletop",
            ]
        )
        self.num_scenario_labels = len(self.scenario_labels)
        self.semantic_loss_weight = float(semantic_loss_weight)
        self.duration_loss_weight = float(duration_loss_weight)
        self.scenario_loss_weight = float(scenario_loss_weight)
        self.knowledge_file = str(knowledge_file)
        self.prompt_template_file = str(prompt_template_file)
        self.knowledge_top_k = int(knowledge_top_k)
        self.max_knowledge_chars = int(max_knowledge_chars)

        self._validate()

    def _validate(self) -> None:
        if not self.base_model_config.get("model_type"):
            raise ValueError("base_model_config must include a non-empty 'model_type'")
        if self.semantic_projection_dim < 1:
            raise ValueError("semantic_projection_dim must be positive")
        if not self.duration_bins or any(value <= 0 for value in self.duration_bins):
            raise ValueError("duration_bins must contain positive minute values")
        if self.duration_bins != sorted(set(self.duration_bins)):
            raise ValueError("duration_bins must be unique and sorted in ascending order")
        if not self.scenario_labels or len(set(self.scenario_labels)) != len(self.scenario_labels):
            raise ValueError("scenario_labels must be non-empty and unique")
        for name in (
            "semantic_loss_weight",
            "duration_loss_weight",
            "scenario_loss_weight",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.knowledge_top_k < 0:
            raise ValueError("knowledge_top_k must be zero or greater")
        if self.max_knowledge_chars < 256:
            raise ValueError("max_knowledge_chars must be at least 256")

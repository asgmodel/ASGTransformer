"""ASGTransformer public Python API."""

from asg_transformer.hf import (
    ASGTransformerCausalLMOutput,
    ASGTransformerConfig,
    ASGTransformerForCausalLM,
    ASGTransformerPreTrainedModel,
)

# Concise direct-construction alias. Hugging Face Auto Classes resolve the full
# ASGTransformerForCausalLM architecture declared in config.json.
ASGTransformer = ASGTransformerForCausalLM

__version__ = "7.0.0"

__all__ = [
    "ASGTransformer",
    "ASGTransformerConfig",
    "ASGTransformerPreTrainedModel",
    "ASGTransformerForCausalLM",
    "ASGTransformerCausalLMOutput",
    "__version__",
]

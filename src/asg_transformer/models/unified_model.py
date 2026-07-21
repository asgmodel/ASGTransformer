"""Backward-compatible imports for ASG Transformer v4 users.

The official model class is now :class:`ASGTransformer`.
"""

from asg_transformer.models.asg_transformer import (
    ASGTransformer,
    ASGTransformerConfig,
    ASGTransformerOutput,
    ASGUnifiedModel,
    UnifiedScenarioOutput,
)

__all__ = [
    "ASGTransformer",
    "ASGTransformerConfig",
    "ASGTransformerOutput",
    "ASGUnifiedModel",
    "UnifiedScenarioOutput",
]

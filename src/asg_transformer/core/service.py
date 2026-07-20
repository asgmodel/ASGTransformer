from __future__ import annotations
from functools import lru_cache
from asg_transformer.config import settings
from asg_transformer.core.catalog import KnowledgeCatalog
from asg_transformer.models.semantic_encoder import SemanticEncoder
from asg_transformer.models.scenario_generator import TransformerScenarioGenerator

class ASGTransformerService:
    def __init__(self):
        self.catalog = KnowledgeCatalog(settings.data_dir)
        reranker = settings.reranker_name if settings.enable_reranker else None
        self.encoder = SemanticEncoder(settings.model_name, str(settings.model_dir), settings.device, reranker)
        self.generator = TransformerScenarioGenerator(self.catalog, self.encoder)

    def classify(self, text: str, task: str, top_k: int):
        collections = {"technique": self.catalog.techniques, "software": self.catalog.software, "group": self.catalog.groups}
        if task not in collections:
            raise ValueError(f"Unsupported task: {task}")
        return self.encoder.rank(text, collections[task], task, top_k)

@lru_cache(maxsize=1)
def get_service() -> ASGTransformerService:
    return ASGTransformerService()

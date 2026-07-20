from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer
from asg_transformer.core.catalog import CatalogItem

@dataclass(slots=True)
class RankedItem:
    item: CatalogItem
    score: float

class SemanticEncoder:
    def __init__(self, model_name: str, model_dir: str, device: str = "auto", reranker_name: str | None = None):
        source = model_dir if __import__("pathlib").Path(model_dir).exists() else model_name
        self.model = SentenceTransformer(source, device=None if device == "auto" else device)
        self.reranker = CrossEncoder(reranker_name) if reranker_name else None
        self._cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _document(item: CatalogItem) -> str:
        tactic = f" Tactic: {item.tactic}." if item.tactic else ""
        return f"Label: {item.label}.{tactic} Description: {item.description}"

    def encode_catalog(self, key: str, items: list[CatalogItem]) -> np.ndarray:
        if key not in self._cache:
            docs = [self._document(item) for item in items]
            self._cache[key] = self.model.encode(docs, normalize_embeddings=True, convert_to_numpy=True)
        return self._cache[key]

    def rank(self, text: str, items: list[CatalogItem], key: str, top_k: int) -> list[RankedItem]:
        matrix = self.encode_catalog(key, items)
        query = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        scores = matrix @ query
        count = min(top_k, len(items))
        idx = np.argpartition(-scores, count - 1)[:count]
        idx = idx[np.argsort(-scores[idx])]
        ranked = [RankedItem(items[int(i)], float(scores[int(i)])) for i in idx]
        if self.reranker and ranked:
            pairs = [(text, self._document(x.item)) for x in ranked]
            reranked_scores = self.reranker.predict(pairs)
            ranked = [RankedItem(x.item, float(s)) for x, s in zip(ranked, reranked_scores)]
            ranked.sort(key=lambda x: x.score, reverse=True)
        return ranked

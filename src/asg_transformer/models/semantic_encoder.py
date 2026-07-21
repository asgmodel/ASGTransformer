from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from asg_transformer.core.catalog import CatalogItem


@dataclass(slots=True)
class RankedItem:
    item: CatalogItem
    score: float


class SemanticEncoder:
    """Sentence-Transformer semantic retrieval with optional cross-encoder reranking.

    Heavy ML dependencies are imported lazily so configuration, documentation,
    and lightweight tests can import the package without initializing PyTorch.
    """

    def __init__(
        self,
        model_name: str,
        model_dir: str,
        device: str = "auto",
        reranker_name: str | None = None,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder, SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "SemanticEncoder requires sentence-transformers. "
                "Install the project with: pip install -e '.[train]'"
            ) from exc

        source = model_dir if Path(model_dir).exists() else model_name
        resolved_device = None if device == "auto" else device
        self.model: Any = SentenceTransformer(source, device=resolved_device)
        self.reranker: Any | None = (
            CrossEncoder(reranker_name, device=resolved_device) if reranker_name else None
        )
        self._cache: dict[str, np.ndarray] = {}

    @staticmethod
    def _document(item: CatalogItem) -> str:
        tactic = f" Tactic: {item.tactic}." if item.tactic else ""
        description = item.description or ""
        return f"Label: {item.label}.{tactic} Description: {description}".strip()

    def encode_catalog(self, key: str, items: list[CatalogItem]) -> np.ndarray:
        if not items:
            return np.empty((0, 0), dtype=np.float32)
        if key not in self._cache:
            documents = [self._document(item) for item in items]
            self._cache[key] = self.model.encode(
                documents,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return self._cache[key]

    def rank(
        self,
        text: str,
        items: list[CatalogItem],
        key: str,
        top_k: int,
    ) -> list[RankedItem]:
        if not text.strip() or not items or top_k <= 0:
            return []

        matrix = self.encode_catalog(key, items)
        query = self.model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        scores = matrix @ query
        count = min(top_k, len(items))
        indices = np.argpartition(-scores, count - 1)[:count]
        indices = indices[np.argsort(-scores[indices])]
        ranked = [RankedItem(items[int(index)], float(scores[int(index)])) for index in indices]

        if self.reranker and ranked:
            pairs = [(text, self._document(result.item)) for result in ranked]
            reranked_scores = self.reranker.predict(pairs)
            ranked = [
                RankedItem(result.item, float(score))
                for result, score in zip(ranked, reranked_scores, strict=True)
            ]
            ranked.sort(key=lambda result: result.score, reverse=True)
        return ranked

    def clear_cache(self) -> None:
        self._cache.clear()

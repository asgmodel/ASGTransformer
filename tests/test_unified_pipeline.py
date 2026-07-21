from __future__ import annotations

from pathlib import Path

import numpy as np

from asg_transformer.core.catalog import KnowledgeCatalog
from asg_transformer.models.semantic_encoder import RankedItem
from asg_transformer.models.unified_model import ASGUnifiedModel


class FakeSentenceModel:
    def save_pretrained(self, path: str, safe_serialization: bool = True) -> None:
        target = Path(path)
        target.mkdir(parents=True, exist_ok=True)
        (target / "modules.json").write_text("{}", encoding="utf-8")


class FakeEncoder:
    def __init__(self) -> None:
        self.model = FakeSentenceModel()

    def rank(self, text, items, key, top_k):
        scores = np.linspace(0.95, 0.55, min(top_k, len(items)))
        return [RankedItem(item, float(score)) for item, score in zip(items[:top_k], scores)]


def test_unified_model_returns_professional_text(tmp_path: Path):
    data_dir = Path(__file__).parents[1] / "data" / "processed"
    model = ASGUnifiedModel(KnowledgeCatalog(data_dir), FakeEncoder(), default_max_steps=4)
    output = model.generate("Defensive phishing and credential access exercise", total_duration_minutes=120)
    assert output.generated_text
    assert output.steps
    assert output.total_duration_minutes >= 40
    assert "Executive Summary" in output.generated_text


def test_unified_model_exports_single_package(tmp_path: Path):
    data_dir = Path(__file__).parents[1] / "data" / "processed"
    model = ASGUnifiedModel(KnowledgeCatalog(data_dir), FakeEncoder())
    exported = model.save_pretrained(tmp_path / "model")
    assert (exported / "asg_config.json").exists()
    assert (exported / "encoder" / "modules.json").exists()
    assert (exported / "knowledge" / "techniques.json").exists()
    assert (exported / "README.md").exists()

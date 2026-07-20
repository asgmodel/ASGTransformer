from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True, slots=True)
class CatalogItem:
    label: str
    description: str
    tactic: str | None = None

class KnowledgeCatalog:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.techniques = self._items("techniques.json")
        self.software = self._items("software.json")
        self.groups = self._items("groups.json")
        self.tactics: dict[str, list[str]] = self._load("tactics.json")
        self.technique_to_tactic: dict[str, str] = self._load("technique_to_tactic.json")
        self.transition_scores: dict[str, dict[str, float]] = self._load("transition_scores.json")
        self.tactic_order = {name: i for i, name in enumerate(self.tactics)}

    def _load(self, filename: str):
        path = self.data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Required catalog file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    def _items(self, filename: str) -> list[CatalogItem]:
        return [CatalogItem(**item) for item in self._load(filename)]

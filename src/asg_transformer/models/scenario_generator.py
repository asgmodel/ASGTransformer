from __future__ import annotations
from dataclasses import dataclass
from asg_transformer.core.catalog import KnowledgeCatalog
from asg_transformer.models.semantic_encoder import RankedItem, SemanticEncoder

@dataclass(slots=True)
class GeneratedStep:
    tactic: str
    technique: str
    semantic_score: float
    transition_score: float
    combined_score: float

class TransformerScenarioGenerator:
    def __init__(self, catalog: KnowledgeCatalog, encoder: SemanticEncoder):
        self.catalog = catalog
        self.encoder = encoder

    def generate(self, text: str, max_steps: int, beam_width: int, transition_weight: float) -> list[GeneratedStep]:
        candidates = self.encoder.rank(text, self.catalog.techniques, "techniques", max(beam_width * max_steps, 20))
        by_tactic: dict[str, list[RankedItem]] = {}
        for candidate in candidates:
            tactic = candidate.item.tactic or self.catalog.technique_to_tactic.get(candidate.item.label, "unknown")
            by_tactic.setdefault(tactic, []).append(candidate)

        ordered_tactics = sorted(by_tactic, key=lambda t: self.catalog.tactic_order.get(t, 10_000))
        beams: list[tuple[list[GeneratedStep], float]] = [([], 0.0)]
        for tactic in ordered_tactics:
            expanded = []
            for path, total in beams:
                previous = path[-1].technique if path else None
                for candidate in by_tactic[tactic][:beam_width]:
                    transition = self._transition(previous, candidate.item.label)
                    combined = (1.0 - transition_weight) * candidate.score + transition_weight * transition
                    step = GeneratedStep(tactic, candidate.item.label, candidate.score, transition, combined)
                    expanded.append((path + [step], total + combined))
            if expanded:
                beams = sorted(expanded, key=lambda x: x[1] / len(x[0]), reverse=True)[:beam_width]
            if beams and len(beams[0][0]) >= max_steps:
                break
        return beams[0][0] if beams else []

    def _transition(self, previous: str | None, current: str) -> float:
        if previous is None:
            return 1.0
        raw = self.catalog.transition_scores.get(previous, {}).get(current, 0.0)
        return max(0.0, min(float(raw) / 100.0, 1.0))

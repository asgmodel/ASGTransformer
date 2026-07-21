from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

from asg_transformer.models.scenario_generator import GeneratedStep


@dataclass(slots=True)
class PlannedStep:
    order: int
    tactic: str
    technique: str
    duration_minutes: int
    semantic_score: float
    transition_score: float
    combined_score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class DurationPlanner:
    """Deterministic duration and step-budget planner.

    This module converts ranked scenario steps into an operationally readable
    sequence with bounded duration estimates. It is intentionally deterministic
    so exported Hugging Face packages produce reproducible results.
    """

    def __init__(
        self,
        minimum_minutes: int = 10,
        maximum_minutes: int = 120,
        default_total_minutes: int = 180,
    ) -> None:
        if minimum_minutes < 1 or maximum_minutes < minimum_minutes:
            raise ValueError("Invalid duration bounds")
        self.minimum_minutes = minimum_minutes
        self.maximum_minutes = maximum_minutes
        self.default_total_minutes = default_total_minutes

    def plan(
        self,
        steps: Sequence[GeneratedStep],
        requested_total_minutes: int | None = None,
    ) -> list[PlannedStep]:
        if not steps:
            return []

        total_minutes = max(
            self.minimum_minutes * len(steps),
            requested_total_minutes or self.default_total_minutes,
        )
        weights = [self._weight(step) for step in steps]
        weight_sum = sum(weights) or float(len(steps))

        raw = [total_minutes * weight / weight_sum for weight in weights]
        durations = [
            min(self.maximum_minutes, max(self.minimum_minutes, int(round(value / 5.0) * 5)))
            for value in raw
        ]

        return [
            PlannedStep(
                order=index,
                tactic=step.tactic,
                technique=step.technique,
                duration_minutes=durations[index - 1],
                semantic_score=step.semantic_score,
                transition_score=step.transition_score,
                combined_score=step.combined_score,
            )
            for index, step in enumerate(steps, start=1)
        ]

    @staticmethod
    def _weight(step: GeneratedStep) -> float:
        confidence = max(0.05, min(step.combined_score, 1.0))
        complexity = 1.0 + 0.25 * (1.0 - step.transition_score)
        return math.sqrt(confidence) * complexity

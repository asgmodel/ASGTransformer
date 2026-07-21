from __future__ import annotations

from pydantic import BaseModel, Field


class ScenarioRequest(BaseModel):
    text: str = Field(min_length=3, max_length=20_000)
    language: str = Field(default="en", pattern="^(en|ar)$")
    max_new_tokens: int | None = Field(default=None, ge=16, le=4096)
    do_sample: bool = False
    temperature: float = Field(default=0.7, gt=0.0, le=2.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)


class KnowledgeMatch(BaseModel):
    source: str
    text: str
    score: float


class ScenarioResponse(BaseModel):
    input_text: str
    generated_text: str
    estimated_duration_minutes: int
    duration_probabilities: list[float]
    scenario_type: str
    scenario_probabilities: list[float]
    knowledge_matches: list[KnowledgeMatch]


class EmbeddingRequest(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class EmbeddingResponse(BaseModel):
    dimension: int
    embedding: list[float]

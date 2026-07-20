from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    text: str = Field(min_length=3, max_length=10_000)
    top_k: int | None = Field(default=None, ge=1, le=50)

class Candidate(BaseModel):
    label: str
    score: float
    description: str | None = None
    tactic: str | None = None

class ClassificationResponse(BaseModel):
    task: str
    candidates: list[Candidate]

class ScenarioRequest(PredictionRequest):
    max_steps: int | None = Field(default=None, ge=2, le=20)
    beam_width: int = Field(default=5, ge=1, le=20)
    transition_weight: float = Field(default=0.35, ge=0.0, le=1.0)

class ScenarioStep(BaseModel):
    order: int
    tactic: str
    technique: str
    semantic_score: float
    transition_score: float
    combined_score: float

class ScenarioResponse(BaseModel):
    input_text: str
    confidence: float
    steps: list[ScenarioStep]
    related_software: list[Candidate]
    related_groups: list[Candidate]

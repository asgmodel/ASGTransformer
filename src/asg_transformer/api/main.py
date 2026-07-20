from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import ORJSONResponse
from asg_transformer import __version__
from asg_transformer.config import settings
from asg_transformer.core.schemas import Candidate, ClassificationResponse, PredictionRequest, ScenarioRequest, ScenarioResponse, ScenarioStep
from asg_transformer.core.service import get_service

@asynccontextmanager
async def lifespan(_: FastAPI):
    get_service()
    yield

app = FastAPI(title="ASG Transformer API", version=__version__, default_response_class=ORJSONResponse, lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "healthy", "version": __version__}

@app.post("/v1/classify/{task}", response_model=ClassificationResponse)
def classify(task: str, request: PredictionRequest):
    try:
        ranked = get_service().classify(request.text, task, request.top_k or settings.top_k)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return ClassificationResponse(task=task, candidates=[Candidate(label=x.item.label, score=x.score, description=x.item.description, tactic=x.item.tactic) for x in ranked])

@app.post("/v1/scenarios/generate", response_model=ScenarioResponse)
def generate_scenario(request: ScenarioRequest):
    service = get_service()
    steps = service.generator.generate(request.text, request.max_steps or settings.max_scenario_steps, request.beam_width, request.transition_weight)
    software = service.classify(request.text, "software", 3)
    groups = service.classify(request.text, "group", 3)
    confidence = sum(x.combined_score for x in steps) / len(steps) if steps else 0.0
    return ScenarioResponse(
        input_text=request.text, confidence=confidence,
        steps=[ScenarioStep(order=i+1, tactic=x.tactic, technique=x.technique, semantic_score=x.semantic_score, transition_score=x.transition_score, combined_score=x.combined_score) for i,x in enumerate(steps)],
        related_software=[Candidate(label=x.item.label, score=x.score, description=x.item.description) for x in software],
        related_groups=[Candidate(label=x.item.label, score=x.score, description=x.item.description) for x in groups],
    )

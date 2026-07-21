from contextlib import asynccontextmanager
import time
import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import ORJSONResponse
from asg_transformer import __version__
from asg_transformer.config import settings
from asg_transformer.core.schemas import Candidate, ClassificationResponse, PredictionRequest, ScenarioRequest, ScenarioResponse, ScenarioStep
from asg_transformer.core.service import get_service
from asg_transformer.observability import configure_logging, metrics

configure_logging(settings.log_level)
log=structlog.get_logger(__name__)

@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info("application_starting",version=__version__,data_dir=str(settings.data_dir),model_dir=str(settings.model_dir))
    get_service(); yield
    log.info("application_stopped")

app=FastAPI(title="ASG Transformer API",description="Transformer-based cyber threat intelligence and scenario generation.",version=__version__,default_response_class=ORJSONResponse,lifespan=lifespan)

@app.middleware("http")
async def observe(request: Request, call_next):
    start=time.perf_counter()
    try: response=await call_next(request)
    except Exception:
        metrics.record(request.url.path,(time.perf_counter()-start)*1000,500); log.exception("request_failed",path=request.url.path); raise
    latency=(time.perf_counter()-start)*1000; metrics.record(request.url.path,latency,response.status_code)
    response.headers["X-Process-Time-Ms"]=f"{latency:.2f}"; return response

@app.get("/")
def root(): return {"name":"ASG Transformer","version":__version__,"docs":"/docs"}
@app.get("/health")
def health(): return {"status":"healthy","version":__version__,"data_ready":settings.data_dir.exists(),"model_ready":settings.model_dir.exists()}
@app.get("/metrics")
def runtime_metrics(): return metrics.snapshot()
@app.post("/v1/classify/{task}",response_model=ClassificationResponse)
def classify(task:str,request:PredictionRequest):
    try: ranked=get_service().classify(request.text,task,request.top_k or settings.top_k)
    except ValueError as exc: raise HTTPException(status_code=400,detail=str(exc)) from exc
    return ClassificationResponse(task=task,candidates=[Candidate(label=x.item.label,score=x.score,description=x.item.description,tactic=x.item.tactic) for x in ranked])
@app.post("/v1/scenarios/generate", response_model=ScenarioResponse)
def generate_scenario(request: ScenarioRequest):
    result = get_service().generate_text(
        request.text,
        max_steps=request.max_steps or settings.max_scenario_steps,
        beam_width=request.beam_width,
        transition_weight=request.transition_weight,
        total_duration_minutes=request.total_duration_minutes,
        language=request.language,
    )
    return ScenarioResponse(
        input_text=result.input_text,
        title=result.title,
        executive_summary=result.executive_summary,
        generated_text=result.generated_text,
        confidence=result.confidence,
        total_duration_minutes=result.total_duration_minutes,
        steps=[ScenarioStep(**step) for step in result.steps],
        related_software=[Candidate(**item) for item in result.related_software],
        related_groups=[Candidate(**item) for item in result.related_groups],
    )

from __future__ import annotations

from contextlib import asynccontextmanager
import time

from fastapi import FastAPI, Request
from fastapi.responses import ORJSONResponse
import structlog

from asg_transformer import __version__
from asg_transformer.config import settings
from asg_transformer.core.schemas import (
    EmbeddingRequest,
    EmbeddingResponse,
    ScenarioRequest,
    ScenarioResponse,
)
from asg_transformer.core.service import get_service
from asg_transformer.observability import configure_logging, metrics

configure_logging(settings.log_level)
log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    log.info(
        "application_starting",
        version=__version__,
        model_source=settings.model_source,
    )
    get_service()
    yield
    log.info("application_stopped")


app = FastAPI(
    title="ASGTransformer API",
    description=(
        "Hugging Face-compatible defensive cybersecurity scenario generation, "
        "duration prediction, scenario classification, and semantic encoding."
    ),
    version=__version__,
    default_response_class=ORJSONResponse,
    lifespan=lifespan,
)


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        latency = (time.perf_counter() - start) * 1000
        metrics.record(request.url.path, latency, 500)
        log.exception("request_failed", path=request.url.path)
        raise
    latency = (time.perf_counter() - start) * 1000
    metrics.record(request.url.path, latency, response.status_code)
    response.headers["X-Process-Time-Ms"] = f"{latency:.2f}"
    return response


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "ASGTransformer",
        "version": __version__,
        "model_source": settings.model_source,
        "docs": "/docs",
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "healthy",
        "version": __version__,
        "model_source": settings.model_source,
        "local_model_available": (settings.model_dir / "config.json").is_file(),
    }


@app.get("/metrics")
def runtime_metrics() -> dict[str, object]:
    return metrics.snapshot()


@app.post("/v1/scenarios/generate", response_model=ScenarioResponse)
def generate_scenario(request: ScenarioRequest) -> ScenarioResponse:
    generation_options: dict[str, object] = {
        "language": request.language,
        "max_new_tokens": request.max_new_tokens or settings.max_new_tokens,
        "do_sample": request.do_sample,
    }
    if request.do_sample:
        generation_options.update(
            temperature=request.temperature,
            top_p=request.top_p,
        )
    result = get_service().generate_scenario(request.text, **generation_options)
    return ScenarioResponse(
        input_text=request.text,
        generated_text=result["text"],
        estimated_duration_minutes=result["estimated_duration_minutes"],
        duration_probabilities=result["duration_probabilities"],
        scenario_type=result["scenario_type"],
        scenario_probabilities=result["scenario_probabilities"],
        knowledge_matches=result["knowledge_matches"],
    )


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
def create_embedding(request: EmbeddingRequest) -> EmbeddingResponse:
    embedding = get_service().encode(request.text)
    return EmbeddingResponse(dimension=len(embedding), embedding=embedding)

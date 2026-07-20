from __future__ import annotations

import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any


class WandbTracker(AbstractContextManager["WandbTracker"]):
    """Optional Weights & Biases experiment tracker.

    The project continues to work when W&B is disabled. When enabled, the
    tracker records configuration, evaluation metrics, timing information, and
    model artifacts.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        project: str,
        entity: str | None = None,
        run_name: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        mode: str = "online",
        config: dict[str, Any] | None = None,
    ) -> None:
        self.enabled = enabled
        self.project = project
        self.entity = entity
        self.run_name = run_name
        self.group = group
        self.tags = tags or []
        self.mode = mode
        self.config = config or {}
        self.run: Any | None = None
        self._started_at = 0.0

    def __enter__(self) -> "WandbTracker":
        self._started_at = time.perf_counter()
        if not self.enabled:
            return self

        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError(
                "W&B tracking is enabled, but the 'wandb' package is not installed. "
                "Install training dependencies with: pip install -e '.[train]'"
            ) from exc

        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.run_name,
            group=self.group,
            tags=self.tags,
            mode=self.mode,
            config=self.config,
            job_type="training",
            resume="allow",
        )
        return self

    def log(self, metrics: dict[str, Any], *, step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def log_model(self, model_dir: Path, *, aliases: list[str] | None = None) -> None:
        if self.run is None or not model_dir.exists():
            return

        import wandb

        artifact = wandb.Artifact(
            name="asg-transformer-model",
            type="model",
            description="Fine-tuned ASG semantic Transformer encoder",
            metadata={"source_directory": str(model_dir)},
        )
        artifact.add_dir(str(model_dir))
        self.run.log_artifact(artifact, aliases=aliases or ["latest"])

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if self.run is not None:
            elapsed = time.perf_counter() - self._started_at
            self.run.log({"runtime/total_seconds": elapsed})
            if exc is not None:
                self.run.summary["status"] = "failed"
                self.run.summary["error"] = str(exc)
            else:
                self.run.summary["status"] = "completed"
            self.run.finish(exit_code=1 if exc is not None else 0)
        return False

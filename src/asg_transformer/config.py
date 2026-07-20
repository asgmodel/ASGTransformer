from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# config.py lives at: <project_root>/src/asg_transformer/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Application settings with paths resolved from the project root.

    Relative values supplied through .env are interpreted relative to
    PROJECT_ROOT, not relative to the shell's current working directory.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="ASG_",
        extra="ignore",
    )

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_dir: Path = PROJECT_ROOT / "models/asg-encoder"
    data_dir: Path = PROJECT_ROOT / "data/processed"
    device: str = "auto"
    top_k: int = Field(5, ge=1, le=50)
    max_scenario_steps: int = Field(8, ge=2, le=20)
    min_confidence: float = Field(0.25, ge=0.0, le=1.0)
    enable_reranker: bool = False
    reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    @model_validator(mode="after")
    def resolve_relative_paths(self) -> "Settings":
        self.model_dir = self._resolve_from_root(self.model_dir)
        self.data_dir = self._resolve_from_root(self.data_dir)
        return self

    @staticmethod
    def _resolve_from_root(path: Path) -> Path:
        path = path.expanduser()
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


settings = Settings()

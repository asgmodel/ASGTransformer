from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime settings for the Hugging Face-compatible ASGTransformer service."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="ASG_",
        extra="ignore",
    )

    model_id: str = "asgmodel/ASGTransformer"
    model_dir: Path = PROJECT_ROOT / "models/ASGTransformer"
    data_dir: Path = PROJECT_ROOT / "data/processed"
    trust_remote_code: bool = True
    torch_dtype: str = "auto"
    device_map: str = "auto"
    max_new_tokens: int = Field(384, ge=16, le=4096)
    default_language: str = Field("en", pattern="^(en|ar)$")
    log_level: str = "INFO"

    @model_validator(mode="after")
    def resolve_relative_paths(self) -> "Settings":
        self.model_dir = self._resolve_from_root(self.model_dir)
        self.data_dir = self._resolve_from_root(self.data_dir)
        return self

    @staticmethod
    def _resolve_from_root(path: Path) -> Path:
        path = path.expanduser()
        return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    @property
    def model_source(self) -> str:
        """Prefer a verified local export; otherwise use the configured Hub ID."""
        return str(self.model_dir) if (self.model_dir / "config.json").is_file() else self.model_id


settings = Settings()

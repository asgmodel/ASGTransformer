from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="ASG_", extra="ignore")
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model_dir: Path = Path("models/asg-encoder")
    data_dir: Path = Path("data/processed")
    device: str = "auto"
    top_k: int = Field(5, ge=1, le=50)
    max_scenario_steps: int = Field(8, ge=2, le=20)
    min_confidence: float = Field(0.25, ge=0.0, le=1.0)
    enable_reranker: bool = False
    reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

settings = Settings()

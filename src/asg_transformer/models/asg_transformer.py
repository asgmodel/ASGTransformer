from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from typing import Any, Literal

from huggingface_hub import HfApi, snapshot_download

from asg_transformer.core.catalog import KnowledgeCatalog
from asg_transformer.models.duration_planner import DurationPlanner
from asg_transformer.models.scenario_generator import TransformerScenarioGenerator
from asg_transformer.models.semantic_encoder import SemanticEncoder
from asg_transformer.models.text_generator import GroundedTextGenerator

Language = Literal["en", "ar"]


@dataclass(slots=True, frozen=True)
class ASGTransformerConfig:
    """Serializable configuration for an exported ASGTransformer package."""

    default_max_steps: int = 8
    default_beam_width: int = 5
    default_transition_weight: float = 0.35
    default_total_minutes: int = 180
    encoder_folder: str = "encoder"
    knowledge_folder: str = "knowledge"

    def validate(self) -> None:
        if self.default_max_steps < 1:
            raise ValueError("default_max_steps must be at least 1")
        if self.default_beam_width < 1:
            raise ValueError("default_beam_width must be at least 1")
        if not 0.0 <= self.default_transition_weight <= 1.0:
            raise ValueError("default_transition_weight must be between 0 and 1")
        if self.default_total_minutes < 1:
            raise ValueError("default_total_minutes must be at least 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "architectures": ["ASGTransformer"],
            "model_type": "asg_transformer",
            "pipeline_tag": "text2text-generation",
            "library_name": "asg-transformer",
            **asdict(self),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ASGTransformerConfig":
        config = cls(
            default_max_steps=int(value.get("default_max_steps", 8)),
            default_beam_width=int(value.get("default_beam_width", 5)),
            default_transition_weight=float(value.get("default_transition_weight", 0.35)),
            default_total_minutes=int(value.get("default_total_minutes", 180)),
            encoder_folder=str(value.get("encoder_folder", "encoder")),
            knowledge_folder=str(value.get("knowledge_folder", "knowledge")),
        )
        config.validate()
        return config


@dataclass(slots=True)
class ASGTransformerOutput:
    """Structured result returned by :meth:`ASGTransformer.generate`."""

    input_text: str
    title: str
    executive_summary: str
    generated_text: str
    confidence: float
    total_duration_minutes: int
    steps: list[dict[str, Any]]
    related_software: list[dict[str, Any]]
    related_groups: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ASGTransformer:
    """Unified, Hugging Face-ready ASG scenario generation model.

    Processing pipeline::

        input text
            -> semantic encoder
            -> catalog-grounded scenario planner
            -> duration planner
            -> professional grounded text generator

    The exported package bundles the encoder, configuration, and knowledge
    catalog. It can therefore be loaded from either a local directory or a
    Hugging Face model repository through :meth:`from_pretrained`.
    """

    CONFIG_NAME = "asg_config.json"

    def __init__(
        self,
        catalog: KnowledgeCatalog,
        encoder: SemanticEncoder,
        config: ASGTransformerConfig | None = None,
        *,
        default_max_steps: int | None = None,
        default_beam_width: int | None = None,
        default_transition_weight: float | None = None,
        default_total_minutes: int | None = None,
    ) -> None:
        base = config or ASGTransformerConfig()
        self.config = ASGTransformerConfig(
            default_max_steps=(
                base.default_max_steps if default_max_steps is None else default_max_steps
            ),
            default_beam_width=(
                base.default_beam_width if default_beam_width is None else default_beam_width
            ),
            default_transition_weight=(
                base.default_transition_weight
                if default_transition_weight is None
                else default_transition_weight
            ),
            default_total_minutes=(
                base.default_total_minutes
                if default_total_minutes is None
                else default_total_minutes
            ),
            encoder_folder=base.encoder_folder,
            knowledge_folder=base.knowledge_folder,
        )
        self.config.validate()
        self.catalog = catalog
        self.encoder = encoder
        self.scenario_generator = TransformerScenarioGenerator(catalog, encoder)
        self.duration_planner = DurationPlanner(
            default_total_minutes=self.config.default_total_minutes
        )
        self.text_generator = GroundedTextGenerator(catalog)

    def __call__(self, text: str, **kwargs: Any) -> ASGTransformerOutput:
        return self.generate(text, **kwargs)

    def generate(
        self,
        text: str,
        *,
        max_steps: int | None = None,
        beam_width: int | None = None,
        transition_weight: float | None = None,
        total_duration_minutes: int | None = None,
        language: Language | str = "en",
    ) -> ASGTransformerOutput:
        """Generate grounded professional text and structured scenario metadata."""
        normalized_text = text.strip()
        if len(normalized_text) < 3:
            raise ValueError("Input text must contain at least three characters")

        resolved_max_steps = max_steps or self.config.default_max_steps
        resolved_beam_width = beam_width or self.config.default_beam_width
        resolved_weight = (
            self.config.default_transition_weight
            if transition_weight is None
            else transition_weight
        )
        resolved_duration = total_duration_minutes or self.config.default_total_minutes

        if resolved_max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if resolved_beam_width < 1:
            raise ValueError("beam_width must be at least 1")
        if not 0.0 <= resolved_weight <= 1.0:
            raise ValueError("transition_weight must be between 0 and 1")
        if resolved_duration < 1:
            raise ValueError("total_duration_minutes must be at least 1")

        raw_steps = self.scenario_generator.generate(
            text=normalized_text,
            max_steps=resolved_max_steps,
            beam_width=resolved_beam_width,
            transition_weight=resolved_weight,
        )
        planned_steps = self.duration_planner.plan(
            raw_steps,
            requested_total_minutes=resolved_duration,
        )
        software = self.encoder.rank(normalized_text, self.catalog.software, "software", 3)
        groups = self.encoder.rank(normalized_text, self.catalog.groups, "groups", 3)
        rendered = self.text_generator.generate(
            normalized_text,
            planned_steps,
            software,
            groups,
            language,
        )
        confidence = (
            sum(step.combined_score for step in planned_steps) / len(planned_steps)
            if planned_steps
            else 0.0
        )

        return ASGTransformerOutput(
            input_text=normalized_text,
            title=rendered.title,
            executive_summary=rendered.executive_summary,
            generated_text=rendered.scenario_text,
            confidence=confidence,
            total_duration_minutes=sum(step.duration_minutes for step in planned_steps),
            steps=[step.to_dict() for step in planned_steps],
            related_software=[
                {
                    "label": ranked.item.label,
                    "score": ranked.score,
                    "description": ranked.item.description,
                }
                for ranked in software
            ],
            related_groups=[
                {
                    "label": ranked.item.label,
                    "score": ranked.score,
                    "description": ranked.item.description,
                }
                for ranked in groups
            ],
        )

    def save_pretrained(self, output_dir: str | Path) -> Path:
        """Export encoder, catalog, runtime files, and model card to one folder."""
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        encoder_dir = output_path / self.config.encoder_folder
        self.encoder.model.save_pretrained(str(encoder_dir), safe_serialization=True)

        knowledge_dir = output_path / self.config.knowledge_folder
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(self.catalog.data_dir.glob("*.json")):
            shutil.copy2(source, knowledge_dir / source.name)

        (output_path / self.CONFIG_NAME).write_text(
            json.dumps(self.config.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self._write_model_card(output_path)
        self._write_hub_runtime(output_path)
        return output_path

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str | Path,
        *,
        device: str = "auto",
        token: str | None = None,
        revision: str | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
    ) -> "ASGTransformer":
        """Load ASGTransformer from a local directory or Hugging Face Hub."""
        path = Path(model_name_or_path).expanduser()
        if not path.exists():
            path = Path(
                snapshot_download(
                    repo_id=str(model_name_or_path),
                    token=token,
                    revision=revision,
                    cache_dir=str(cache_dir) if cache_dir else None,
                    local_files_only=local_files_only,
                )
            )
        path = path.resolve()

        config_path = path / cls.CONFIG_NAME
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing ASGTransformer configuration: {config_path}")

        config = ASGTransformerConfig.from_dict(
            json.loads(config_path.read_text(encoding="utf-8"))
        )
        catalog = KnowledgeCatalog(path / config.knowledge_folder)
        encoder_path = path / config.encoder_folder
        if not encoder_path.is_dir():
            raise FileNotFoundError(f"Missing ASGTransformer encoder: {encoder_path}")
        encoder = SemanticEncoder(
            model_name=str(encoder_path),
            model_dir=str(encoder_path),
            device=device,
        )
        return cls(catalog=catalog, encoder=encoder, config=config)

    def push_to_hub(
        self,
        repo_id: str,
        *,
        private: bool = False,
        token: str | None = None,
        commit_message: str = "Publish ASGTransformer model",
    ) -> str:
        """Export and upload one complete ASGTransformer model repository."""
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        with TemporaryDirectory(prefix="asg-transformer-") as temporary_dir:
            export_dir = self.save_pretrained(temporary_dir)
            api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=str(export_dir),
                commit_message=commit_message,
            )
        return f"https://huggingface.co/{repo_id}"

    @staticmethod
    def _write_hub_runtime(output_path: Path) -> None:
        inference = '''from __future__ import annotations

from typing import Any
from asg_transformer import ASGTransformer

_model: ASGTransformer | None = None


def load_model(model_dir: str = ".") -> ASGTransformer:
    global _model
    if _model is None:
        _model = ASGTransformer.from_pretrained(model_dir)
    return _model


def predict(inputs: dict[str, Any]) -> dict[str, Any]:
    model = load_model()
    text = inputs.get("text") or inputs.get("inputs")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("A non-empty 'text' or 'inputs' field is required")
    return model.generate(
        text,
        max_steps=inputs.get("max_steps"),
        beam_width=inputs.get("beam_width"),
        transition_weight=inputs.get("transition_weight"),
        total_duration_minutes=inputs.get("total_duration_minutes"),
        language=inputs.get("language", "en"),
    ).to_dict()
'''
        (output_path / "inference.py").write_text(inference, encoding="utf-8")
        (output_path / "requirements.txt").write_text(
            "asg-transformer>=5.0.0\nsentence-transformers>=3.3\nhuggingface-hub>=0.27\n",
            encoding="utf-8",
        )

    @staticmethod
    def _write_model_card(output_path: Path) -> None:
        card = '''---
library_name: asg-transformer
pipeline_tag: text2text-generation
tags:
- cybersecurity
- sentence-transformers
- grounded-generation
- scenario-generation
- defensive-security
language:
- en
- ar
license: mit
---

# ASGTransformer

`ASGTransformer` is a unified, catalog-grounded defensive cybersecurity scenario
model. It bundles the semantic encoder, scenario planner, duration planner,
professional text renderer, and knowledge catalog in one Hugging Face repository.

## Pipeline

`Input Text -> Encoder -> Scenario Planner -> Duration Planner -> Text Generator`

## Usage

```python
from asg_transformer import ASGTransformer

model = ASGTransformer.from_pretrained("asgmodel/ASGTransformer")
result = model.generate(
    "Create an authorized defensive enterprise scenario focused on phishing and credential access.",
    language="en",
)
print(result.generated_text)
```

The model is intended for authorized defensive training, tabletop exercises,
detection engineering, control validation, and incident-response preparation.
'''
        (output_path / "README.md").write_text(card, encoding="utf-8")


# Backward compatibility with v4 exports.
ASGUnifiedModel = ASGTransformer
UnifiedScenarioOutput = ASGTransformerOutput

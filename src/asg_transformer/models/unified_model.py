from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shutil
from typing import Any

from huggingface_hub import HfApi

from asg_transformer.core.catalog import KnowledgeCatalog
from asg_transformer.models.duration_planner import DurationPlanner, PlannedStep
from asg_transformer.models.scenario_generator import TransformerScenarioGenerator
from asg_transformer.models.semantic_encoder import SemanticEncoder
from asg_transformer.models.text_generator import GroundedTextGenerator


@dataclass(slots=True)
class UnifiedScenarioOutput:
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


class ASGUnifiedModel:
    """Single deployable ASG model package.

    Pipeline:
        input text -> semantic encoder -> scenario beam planner -> duration planner
        -> grounded professional text generator

    The package follows a Hugging Face-style interface with save_pretrained,
    from_pretrained and push_to_hub methods. The knowledge catalog is bundled
    inside the exported model directory.
    """

    CONFIG_NAME = "asg_config.json"
    DATA_FOLDER = "knowledge"
    ENCODER_FOLDER = "encoder"

    def __init__(
        self,
        catalog: KnowledgeCatalog,
        encoder: SemanticEncoder,
        default_max_steps: int = 8,
        default_beam_width: int = 5,
        default_transition_weight: float = 0.35,
        default_total_minutes: int = 180,
    ) -> None:
        self.catalog = catalog
        self.encoder = encoder
        self.scenario_generator = TransformerScenarioGenerator(catalog, encoder)
        self.duration_planner = DurationPlanner(default_total_minutes=default_total_minutes)
        self.text_generator = GroundedTextGenerator(catalog)
        self.default_max_steps = default_max_steps
        self.default_beam_width = default_beam_width
        self.default_transition_weight = default_transition_weight
        self.default_total_minutes = default_total_minutes

    def __call__(self, text: str, **kwargs: Any) -> UnifiedScenarioOutput:
        return self.generate(text, **kwargs)

    def generate(
        self,
        text: str,
        max_steps: int | None = None,
        beam_width: int | None = None,
        transition_weight: float | None = None,
        total_duration_minutes: int | None = None,
        language: str = "en",
    ) -> UnifiedScenarioOutput:
        text = text.strip()
        if len(text) < 3:
            raise ValueError("Input text must contain at least three characters")

        raw_steps = self.scenario_generator.generate(
            text=text,
            max_steps=max_steps or self.default_max_steps,
            beam_width=beam_width or self.default_beam_width,
            transition_weight=(
                self.default_transition_weight if transition_weight is None else transition_weight
            ),
        )
        planned_steps = self.duration_planner.plan(
            raw_steps,
            requested_total_minutes=total_duration_minutes or self.default_total_minutes,
        )
        software = self.encoder.rank(text, self.catalog.software, "software", 3)
        groups = self.encoder.rank(text, self.catalog.groups, "groups", 3)
        rendered = self.text_generator.generate(text, planned_steps, software, groups, language)
        confidence = (
            sum(step.combined_score for step in planned_steps) / len(planned_steps)
            if planned_steps
            else 0.0
        )
        return UnifiedScenarioOutput(
            input_text=text,
            title=rendered.title,
            executive_summary=rendered.executive_summary,
            generated_text=rendered.scenario_text,
            confidence=confidence,
            total_duration_minutes=sum(step.duration_minutes for step in planned_steps),
            steps=[step.to_dict() for step in planned_steps],
            related_software=[
                {"label": item.item.label, "score": item.score, "description": item.item.description}
                for item in software
            ],
            related_groups=[
                {"label": item.item.label, "score": item.score, "description": item.item.description}
                for item in groups
            ],
        )

    def save_pretrained(self, output_dir: str | Path) -> Path:
        output_path = Path(output_dir).expanduser().resolve()
        output_path.mkdir(parents=True, exist_ok=True)

        encoder_dir = output_path / self.ENCODER_FOLDER
        self.encoder.model.save_pretrained(str(encoder_dir), safe_serialization=True)

        knowledge_dir = output_path / self.DATA_FOLDER
        knowledge_dir.mkdir(parents=True, exist_ok=True)
        for source in self.catalog.data_dir.glob("*.json"):
            shutil.copy2(source, knowledge_dir / source.name)

        config = {
            "architectures": ["ASGUnifiedModel"],
            "pipeline_tag": "text2text-generation",
            "library_name": "asg-transformer",
            "default_max_steps": self.default_max_steps,
            "default_beam_width": self.default_beam_width,
            "default_transition_weight": self.default_transition_weight,
            "default_total_minutes": self.default_total_minutes,
            "encoder_folder": self.ENCODER_FOLDER,
            "knowledge_folder": self.DATA_FOLDER,
        }
        (output_path / self.CONFIG_NAME).write_text(
            json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._write_model_card(output_path)
        self._write_hub_runtime(output_path)
        return output_path

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str = "auto",
        token: str | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
    ) -> "ASGUnifiedModel":
        path = Path(model_name_or_path)
        if not path.exists():
            from huggingface_hub import snapshot_download

            path = Path(
                snapshot_download(
                    repo_id=model_name_or_path,
                    token=token,
                    revision=revision,
                    cache_dir=cache_dir,
                )
            )

        config_path = path / cls.CONFIG_NAME
        if not config_path.exists():
            raise FileNotFoundError(f"Missing unified model configuration: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        catalog = KnowledgeCatalog(path / config.get("knowledge_folder", cls.DATA_FOLDER))
        encoder_path = path / config.get("encoder_folder", cls.ENCODER_FOLDER)
        encoder = SemanticEncoder(
            model_name=str(encoder_path),
            model_dir=str(encoder_path),
            device=device,
        )
        return cls(
            catalog=catalog,
            encoder=encoder,
            default_max_steps=int(config.get("default_max_steps", 8)),
            default_beam_width=int(config.get("default_beam_width", 5)),
            default_transition_weight=float(config.get("default_transition_weight", 0.35)),
            default_total_minutes=int(config.get("default_total_minutes", 180)),
        )

    def push_to_hub(
        self,
        repo_id: str,
        private: bool = False,
        token: str | None = None,
        commit_message: str = "Publish ASG unified model",
    ) -> str:
        export_dir = Path(".asg_hub_export") / repo_id.replace("/", "__")
        if export_dir.exists():
            shutil.rmtree(export_dir)
        self.save_pretrained(export_dir)
        api = HfApi(token=token)
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(export_dir),
            commit_message=commit_message,
        )
        return f"https://huggingface.co/{repo_id}"

    @staticmethod
    def _write_hub_runtime(output_path: Path) -> None:
        inference = """from asg_transformer import ASGUnifiedModel

_model = None

def load_model(model_dir: str = "."):
    global _model
    if _model is None:
        _model = ASGUnifiedModel.from_pretrained(model_dir)
    return _model

def predict(inputs: dict):
    model = load_model()
    text = inputs.get("text") or inputs.get("inputs")
    if not text:
        raise ValueError("A text or inputs field is required")
    result = model.generate(
        text,
        max_steps=inputs.get("max_steps"),
        total_duration_minutes=inputs.get("total_duration_minutes"),
        language=inputs.get("language", "en"),
    )
    return result.to_dict()
"""
        (output_path / "inference.py").write_text(inference, encoding="utf-8")
        requirements = "asg-transformer>=4.0.0\nsentence-transformers>=3.3\nhuggingface-hub>=0.27\n"
        (output_path / "requirements.txt").write_text(requirements, encoding="utf-8")

    @staticmethod
    def _write_model_card(output_path: Path) -> None:
        card = """---
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

# ASG Unified Transformer Scenario Model

A unified, catalog-grounded defensive cybersecurity scenario model.

## Pipeline

`Input Text -> Semantic Encoder -> Scenario Planner -> Duration Planner -> Grounded Text Generator`

## Usage

```python
from asg_transformer import ASGUnifiedModel

model = ASGUnifiedModel.from_pretrained("asgmodel/ASG-Unified-Scenario-Model")
result = model.generate(
    "Create a defensive enterprise scenario focused on phishing and credential access.",
    language="en",
)
print(result.generated_text)
```

The model is intended for authorized defensive training, tabletop exercises,
detection engineering, and incident-response preparation.
"""
        (output_path / "README.md").write_text(card, encoding="utf-8")
